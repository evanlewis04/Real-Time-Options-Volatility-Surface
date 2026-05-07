"""Persisted-snapshot surface change and vol-of-vol analytics."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.snapshots import list_snapshots, load_snapshot
from src.quant.iv_history import atm_iv_from_chain


def surface_change_analytics(
    symbol: str,
    current_chain: pd.DataFrame,
    spot: float,
    directory: Path | str,
    *,
    iv_column: str = "computedIV",
    current_timestamp: datetime | pd.Timestamp | None = None,
    vol_of_vol_history: list[dict[str, Any]] | None = None,
    min_points: int = 1,
) -> dict[str, Any]:
    """Compare current option IVs with the latest earlier persisted snapshot."""
    current = _prepared_iv_frame(current_chain, iv_column)
    if current.empty:
        return _unavailable("Current surface has no usable IV rows")

    previous_snapshot = None
    previous_frame = pd.DataFrame()
    as_of = _timestamp_or_none(current_timestamp)
    for metadata_path in list_snapshots(symbol, directory):
        try:
            candidate = load_snapshot(metadata_path)
        except Exception:
            continue
        if as_of is not None and candidate.spot_timestamp >= as_of:
            continue
        candidate_frame = _prepared_iv_frame(candidate.options_frame(), iv_column)
        if candidate_frame.empty and iv_column != "computedIV":
            candidate_frame = _prepared_iv_frame(candidate.options_frame(), "computedIV")
        if candidate_frame.empty:
            continue
        previous_snapshot = candidate
        previous_frame = candidate_frame
        break

    if previous_snapshot is None or previous_frame.empty:
        return _unavailable("No earlier persisted snapshot with usable IV rows")

    matched = current.merge(
        previous_frame,
        on=["expiration_key", "strike_key", "type_key"],
        suffixes=("_current", "_previous"),
    )
    if len(matched) < min_points:
        return _unavailable("No matching expiry/strike/type rows in previous snapshot")

    matched["iv_change"] = matched["iv_current"] - matched["iv_previous"]
    matched["iv_change_pct"] = np.where(
        matched["iv_previous"].abs() > 1e-12,
        matched["iv_change"] / matched["iv_previous"],
        np.nan,
    )
    matched["abs_iv_change"] = matched["iv_change"].abs()

    expiry_changes = _expiry_change_records(matched)
    top_changes = _top_change_records(matched)
    atm = _atm_change(matched, spot)
    if vol_of_vol_history is not None:
        vol_of_vol = atm_iv_vol_of_vol_from_observations(
            vol_of_vol_history,
            current_iv=atm.get("current_atm_iv"),
            current_timestamp=as_of,
        )
    else:
        vol_of_vol = atm_iv_vol_of_vol_from_snapshots(
            symbol,
            directory,
            current_iv=atm.get("current_atm_iv"),
            current_timestamp=as_of,
        )

    return {
        "available": True,
        "source": "persisted_snapshots",
        "previous_snapshot_timestamp": previous_snapshot.spot_timestamp.isoformat(),
        "previous_snapshot_source": previous_snapshot.source,
        "previous_snapshot_mode": previous_snapshot.mode,
        "matched_points": int(len(matched)),
        "mean_iv_change": float(matched["iv_change"].mean()),
        "median_iv_change": float(matched["iv_change"].median()),
        "median_abs_iv_change": float(matched["abs_iv_change"].median()),
        "max_abs_iv_change": float(matched["abs_iv_change"].max()),
        "up_points": int((matched["iv_change"] > 0).sum()),
        "down_points": int((matched["iv_change"] < 0).sum()),
        "unchanged_points": int((matched["iv_change"].abs() <= 1e-12).sum()),
        "atm_change": atm,
        "expiry_changes": expiry_changes,
        "top_changes": top_changes,
        "vol_of_vol": vol_of_vol,
    }


def atm_iv_vol_of_vol_from_snapshots(
    symbol: str,
    directory: Path | str,
    *,
    current_iv: float | None = None,
    current_timestamp: datetime | pd.Timestamp | None = None,
    min_points: int = 3,
) -> dict[str, Any]:
    """Estimate snapshot-to-snapshot ATM-IV volatility from stored snapshots."""
    observations: list[dict[str, Any]] = []
    as_of = _timestamp_or_none(current_timestamp)
    for metadata_path in reversed(list_snapshots(symbol, directory)):
        try:
            snapshot = load_snapshot(metadata_path)
        except Exception:
            continue
        if as_of is not None and snapshot.spot_timestamp >= as_of:
            continue
        iv = atm_iv_from_chain(snapshot.options_frame(), snapshot.spot)
        if iv is None or not np.isfinite(iv):
            continue
        observations.append(
            {
                "timestamp": snapshot.spot_timestamp.isoformat(),
                "atm_iv": float(iv),
                "source": snapshot.source,
                "mode": snapshot.mode,
            }
        )

    return atm_iv_vol_of_vol_from_observations(
        observations,
        current_iv=current_iv,
        current_timestamp=as_of,
        min_points=min_points,
    )


def atm_iv_vol_of_vol_from_observations(
    observations: list[dict[str, Any]],
    *,
    current_iv: float | None = None,
    current_timestamp: datetime | pd.Timestamp | None = None,
    min_points: int = 3,
) -> dict[str, Any]:
    """Estimate snapshot-to-snapshot ATM-IV volatility from prepared history rows."""
    prepared = []
    for item in observations:
        try:
            iv = float(item.get("atm_iv"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(iv) or iv <= 0.0:
            continue
        prepared.append(dict(item, atm_iv=iv))

    prepared.sort(key=lambda item: str(item.get("timestamp") or ""))
    as_of = _timestamp_or_none(current_timestamp)
    if current_iv is not None and np.isfinite(current_iv) and current_iv > 0.0:
        prepared.append(
            {
                "timestamp": as_of.isoformat() if as_of is not None else "current",
                "atm_iv": float(current_iv),
                "source": "current_surface",
                "mode": "Current",
            }
        )

    if len(prepared) < min_points:
        return {
            "available": False,
            "source": "persisted_snapshots",
            "reason": f"Need at least {min_points} ATM-IV observations",
            "observations": len(prepared),
            "snapshot_vol_of_vol": None,
            "annualized_vol_of_vol": None,
            "mean_abs_change": None,
            "history": prepared,
        }

    ivs = np.array([item["atm_iv"] for item in prepared], dtype=float)
    changes = np.diff(ivs)
    if changes.size == 0:
        snapshot_vol = np.nan
        mean_abs_change = np.nan
    else:
        snapshot_vol = float(np.std(changes, ddof=1)) if changes.size > 1 else float(abs(changes[0]))
        mean_abs_change = float(np.mean(np.abs(changes)))

    return {
        "available": bool(np.isfinite(snapshot_vol)),
        "source": "persisted_snapshots",
        "observations": int(len(prepared)),
        "change_observations": int(changes.size),
        "snapshot_vol_of_vol": None if not np.isfinite(snapshot_vol) else snapshot_vol,
        "annualized_vol_of_vol": None if not np.isfinite(snapshot_vol) else float(snapshot_vol * np.sqrt(252.0)),
        "mean_abs_change": None if not np.isfinite(mean_abs_change) else mean_abs_change,
        "latest_atm_iv": float(ivs[-1]),
        "history": prepared[-120:],
    }


def _prepared_iv_frame(chain: pd.DataFrame, iv_column: str) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    column = iv_column if iv_column in chain.columns else "impliedVolatility"
    required = {"expiration", "strike", column}
    if not required.issubset(chain.columns):
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "expiration": pd.to_datetime(chain["expiration"], errors="coerce"),
            "strike": pd.to_numeric(chain["strike"], errors="coerce"),
            "iv": pd.to_numeric(chain[column], errors="coerce"),
            "dte": pd.to_numeric(chain.get("daysToExpiration"), errors="coerce"),
            "type_key": (
                chain["type"].astype(str).str.lower()
                if "type" in chain.columns
                else pd.Series("__all__", index=chain.index)
            ),
        }
    )
    out = out.dropna(subset=["expiration", "strike", "iv"])
    out = out[(out["strike"] > 0.0) & (out["iv"] > 0.0)]
    if out.empty:
        return out
    out["expiration_key"] = out["expiration"].dt.date.astype(str)
    out["strike_key"] = out["strike"].round(6)
    return out.sort_values(["expiration_key", "strike_key", "type_key"]).reset_index(drop=True)


def _expiry_change_records(matched: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for expiry, group in matched.groupby("expiration_key", sort=True):
        records.append(
            {
                "expiration": expiry,
                "matched_points": int(len(group)),
                "current_median_iv": float(group["iv_current"].median()),
                "previous_median_iv": float(group["iv_previous"].median()),
                "mean_iv_change": float(group["iv_change"].mean()),
                "median_iv_change": float(group["iv_change"].median()),
                "median_abs_iv_change": float(group["abs_iv_change"].median()),
                "max_abs_iv_change": float(group["abs_iv_change"].max()),
                "up_points": int((group["iv_change"] > 0).sum()),
                "down_points": int((group["iv_change"] < 0).sum()),
            }
        )
    return records


def _top_change_records(matched: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
    rows = matched.sort_values("abs_iv_change", ascending=False).head(limit)
    records = []
    for _, row in rows.iterrows():
        records.append(
            {
                "expiration": row["expiration_key"],
                "type": row["type_key"],
                "strike": float(row["strike_current"]),
                "current_iv": float(row["iv_current"]),
                "previous_iv": float(row["iv_previous"]),
                "iv_change": float(row["iv_change"]),
                "iv_change_pct": float(row["iv_change_pct"]) if np.isfinite(row["iv_change_pct"]) else None,
            }
        )
    return records


def _atm_change(matched: pd.DataFrame, spot: float) -> dict[str, Any]:
    work = matched.copy()
    work["atm_distance"] = (work["strike_current"] - float(spot)).abs()
    work["dte_distance"] = (pd.to_numeric(work["dte_current"], errors="coerce") - 30.0).abs()
    row = work.sort_values(["dte_distance", "atm_distance"]).iloc[0]
    change_pct = row["iv_change_pct"]
    return {
        "expiration": row["expiration_key"],
        "type": row["type_key"],
        "strike": float(row["strike_current"]),
        "dte": None if pd.isna(row["dte_current"]) else float(row["dte_current"]),
        "current_atm_iv": float(row["iv_current"]),
        "previous_atm_iv": float(row["iv_previous"]),
        "iv_change": float(row["iv_change"]),
        "iv_change_pct": float(change_pct) if np.isfinite(change_pct) else None,
    }


def _timestamp_or_none(value: datetime | pd.Timestamp | None) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "source": "persisted_snapshots",
        "reason": reason,
        "matched_points": 0,
        "atm_change": {},
        "expiry_changes": [],
        "top_changes": [],
        "vol_of_vol": {
            "available": False,
            "source": "persisted_snapshots",
            "reason": reason,
            "observations": 0,
            "snapshot_vol_of_vol": None,
            "annualized_vol_of_vol": None,
            "mean_abs_change": None,
            "history": [],
        },
    }
