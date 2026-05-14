"""Persisted-snapshot surface change and vol-of-vol analytics."""

from __future__ import annotations

from datetime import datetime, timedelta
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

    as_of = _timestamp_or_none(current_timestamp)
    previous_snapshot, previous_frame = _select_baseline_snapshot(
        symbol,
        directory,
        iv_column,
        as_of,
        "previous_refresh",
    )

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
        "heatmaps": surface_change_heatmaps(
            symbol,
            current_chain,
            directory,
            iv_column=iv_column,
            current_timestamp=as_of,
        ),
        "tape": surface_tape_analytics(
            symbol,
            directory,
            current_chain=current_chain,
            iv_column=iv_column,
            current_timestamp=as_of,
        ),
        "vol_of_vol": vol_of_vol,
    }


def surface_change_heatmaps(
    symbol: str,
    current_chain: pd.DataFrame,
    directory: Path | str,
    *,
    iv_column: str = "computedIV",
    current_timestamp: datetime | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Build current-minus-baseline IV heatmap rows for common comparison anchors."""
    current = _prepared_iv_frame(current_chain, iv_column)
    if current.empty:
        return {"available": False, "reason": "Current surface has no usable IV rows", "baselines": {}}

    as_of = _timestamp_or_none(current_timestamp)
    baselines: dict[str, Any] = {}
    for baseline in ("previous_refresh", "previous_hour", "previous_close"):
        snapshot, frame = _select_baseline_snapshot(symbol, directory, iv_column, as_of, baseline)
        label = baseline.replace("_", " ").title()
        if snapshot is None or frame.empty:
            baselines[baseline] = {
                "available": False,
                "label": label,
                "reason": f"No {baseline.replace('_', ' ')} snapshot with usable IV rows",
                "records": [],
            }
            continue
        matched = _matched_change_frame(current, frame)
        if matched.empty:
            baselines[baseline] = {
                "available": False,
                "label": label,
                "reason": "No matching expiry/strike/type rows in baseline snapshot",
                "records": [],
                "baseline_timestamp": snapshot.spot_timestamp.isoformat(),
                "baseline_source": snapshot.source,
                "baseline_mode": snapshot.mode,
            }
            continue
        baselines[baseline] = {
            "available": True,
            "label": label,
            "baseline_timestamp": snapshot.spot_timestamp.isoformat(),
            "baseline_source": snapshot.source,
            "baseline_mode": snapshot.mode,
            "matched_points": int(len(matched)),
            "mean_iv_change": float(matched["iv_change"].mean()),
            "max_abs_iv_change": float(matched["abs_iv_change"].max()),
            "records": _heatmap_records(matched),
        }

    return {
        "available": any(payload.get("available") for payload in baselines.values()),
        "source": "persisted_snapshots",
        "baselines": baselines,
    }


def surface_tape_analytics(
    symbol: str,
    directory: Path | str,
    *,
    current_chain: pd.DataFrame | None = None,
    iv_column: str = "computedIV",
    current_timestamp: datetime | pd.Timestamp | None = None,
    max_snapshots: int = 80,
) -> dict[str, Any]:
    """Return intraday persisted-snapshot tape records suitable for replay."""
    as_of = _timestamp_or_none(current_timestamp)
    loaded: list[dict[str, Any]] = []
    target_day = as_of.date() if as_of is not None else None

    for metadata_path in list_snapshots(symbol, directory):
        try:
            snapshot = load_snapshot(metadata_path)
        except Exception:
            continue
        if as_of is not None and snapshot.spot_timestamp > as_of:
            continue
        if target_day is None:
            target_day = snapshot.spot_timestamp.date()
        if snapshot.spot_timestamp.date() != target_day:
            continue
        frame = _prepared_iv_frame(snapshot.options_frame(), iv_column)
        if frame.empty and iv_column != "computedIV":
            frame = _prepared_iv_frame(snapshot.options_frame(), "computedIV")
        if frame.empty:
            continue
        loaded.append(_tape_record(snapshot.symbol, snapshot.spot_timestamp, snapshot.source, snapshot.mode, frame))

    if current_chain is not None:
        current = _prepared_iv_frame(current_chain, iv_column)
        if not current.empty:
            timestamp = as_of or datetime.now()
            if target_day is None or timestamp.date() == target_day:
                loaded.append(_tape_record(symbol.upper(), timestamp, "current_surface", "Current", current))

    loaded.sort(key=lambda item: item["timestamp"])
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in loaded:
        key = str(item["timestamp"])
        if key in seen:
            deduped[-1] = item
            continue
        seen.add(key)
        deduped.append(item)

    snapshots = deduped[-max_snapshots:]
    return {
        "available": bool(snapshots),
        "source": "persisted_snapshots",
        "symbol": symbol.upper(),
        "snapshot_count": len(snapshots),
        "timestamps": [item["timestamp"] for item in snapshots],
        "snapshots": snapshots,
    }


def surface_shape_change_quality_flag(
    change: dict[str, Any],
    *,
    current_quality_score: float | None = None,
    previous_quality_score: float | None = None,
    current_reason_buckets: dict[str, int] | None = None,
    previous_reason_buckets: dict[str, int] | None = None,
    material_change_threshold: float = 0.02,
) -> dict[str, Any]:
    """Classify whether a surface move is likely data-quality driven."""
    if not change.get("available"):
        return {
            "available": False,
            "reason": "Surface change comparison unavailable",
            "provenance": "surface_change_quality_diagnostic_not_market_observation",
        }

    current_buckets = {str(key): int(value) for key, value in (current_reason_buckets or {}).items()}
    previous_buckets = {str(key): int(value) for key, value in (previous_reason_buckets or {}).items()}
    deteriorated_buckets = {
        key: int(current_buckets.get(key, 0) - previous_buckets.get(key, 0))
        for key in sorted(set(current_buckets) | set(previous_buckets))
        if int(current_buckets.get(key, 0) - previous_buckets.get(key, 0)) > 0
    }
    score_change = None
    quality_deteriorated = False
    if current_quality_score is not None and previous_quality_score is not None:
        score_change = float(current_quality_score) - float(previous_quality_score)
        quality_deteriorated = bool(score_change <= -5.0)
    material_shape_change = bool(float(change.get("max_abs_iv_change") or 0.0) >= material_change_threshold)
    likely_quality_driven = bool(material_shape_change and (quality_deteriorated or deteriorated_buckets))
    if likely_quality_driven:
        reason = "Material shape change coincides with deteriorating quote-quality buckets."
    elif material_shape_change:
        reason = "Material shape change without matching quality deterioration; review as possible real move."
    else:
        reason = "No material shape change."
    return {
        "available": True,
        "provenance": "surface_change_quality_diagnostic_not_market_observation",
        "likely_data_quality_driven": likely_quality_driven,
        "material_shape_change": material_shape_change,
        "max_abs_iv_change": change.get("max_abs_iv_change"),
        "median_abs_iv_change": change.get("median_abs_iv_change"),
        "current_quality_score": current_quality_score,
        "previous_quality_score": previous_quality_score,
        "quality_score_change": score_change,
        "quality_deteriorated": quality_deteriorated,
        "deteriorated_buckets": deteriorated_buckets,
        "reason": reason,
    }


def rich_cheap_scanner(
    chain: pd.DataFrame,
    svi_smiles: list[dict[str, Any]],
    *,
    iv_column: str = "computedIV",
    fit_mode: str = "Robust SVI",
    limit: int = 20,
) -> dict[str, Any]:
    """Rank options by IV residual to fitted SVI surface plus liquidity context."""
    residual_rows = _svi_residual_frame(svi_smiles)
    current = _scanner_chain_frame(chain, iv_column)
    if residual_rows.empty:
        return _scanner_unavailable("SVI residuals are unavailable")
    if current.empty:
        return _scanner_unavailable("Current chain has no usable scanner rows")

    joined = current.merge(residual_rows, on=["expiration_key", "strike_key"], how="inner")
    joined = joined.dropna(subset=["surface_residual"])
    if joined.empty:
        return _scanner_unavailable("No chain rows matched fitted-surface residuals")

    residuals = joined["surface_residual"].to_numpy(dtype=float)
    std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    mean = float(np.mean(residuals))
    if std > 1e-12:
        joined["residual_z_score"] = (joined["surface_residual"] - mean) / std
    else:
        joined["residual_z_score"] = 0.0

    joined["liquidity_score"] = _liquidity_scores(joined)
    joined["quote_reliability_score"] = pd.to_numeric(
        joined.get("quote_reliability_score", 1.0),
        errors="coerce",
    ).fillna(1.0).clip(0.0, 1.0)
    joined["candidate_confidence"] = (joined["liquidity_score"] * joined["quote_reliability_score"]).clip(0.0, 1.0)
    joined["scanner_score"] = joined["residual_z_score"].abs() * joined["candidate_confidence"]
    joined = joined.sort_values(
        ["scanner_score", "liquidity_score", "abs_surface_residual"],
        ascending=[False, False, False],
    )

    candidates = []
    for _, row in joined.head(limit).iterrows():
        direction = "rich" if row["surface_residual"] > 0 else "cheap"
        candidates.append(
            {
                "contract": row.get("contract"),
                "type": row.get("type"),
                "expiration": row["expiration_key"],
                "dte": None if pd.isna(row.get("dte")) else float(row.get("dte")),
                "strike": float(row["strike"]),
                "market_iv": float(row["iv"]),
                "fitted_iv": float(row["fitted_iv"]),
                "surface_residual": float(row["surface_residual"]),
                "abs_surface_residual": float(row["abs_surface_residual"]),
                "residual_z_score": float(row["residual_z_score"]),
                "liquidity_score": float(row["liquidity_score"]),
                "quote_reliability_score": float(row["quote_reliability_score"]),
                "candidate_confidence": float(row["candidate_confidence"]),
                "confidence_label": _confidence_label(float(row["candidate_confidence"])),
                "scanner_score": float(row["scanner_score"]),
                "bid_ask_spread_pct": _finite_or_none(row.get("bid_ask_spread_pct")),
                "volume": _finite_or_none(row.get("volume")),
                "open_interest": _finite_or_none(row.get("open_interest")),
                "classification": direction,
                "reason": _scanner_reason(row, direction),
            }
        )

    return {
        "available": bool(candidates),
        "source": "current_chain_plus_svi_fit",
        "model": "SVI",
        "fit_mode": fit_mode,
        "ranking_policy": "abs_residual_z_score_times_liquidity_and_quote_reliability",
        "input_rows": int(len(joined)),
        "candidate_count": len(candidates),
        "rich_count": int((joined["surface_residual"] > 0).sum()),
        "cheap_count": int((joined["surface_residual"] < 0).sum()),
        "residual_mean": mean,
        "residual_std": std,
        "candidates": candidates,
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


def _select_baseline_snapshot(
    symbol: str,
    directory: Path | str,
    iv_column: str,
    as_of: datetime | None,
    baseline: str,
) -> tuple[Any | None, pd.DataFrame]:
    threshold = None
    if baseline == "previous_hour" and as_of is not None:
        threshold = as_of - timedelta(hours=1)

    for metadata_path in list_snapshots(symbol, directory):
        try:
            candidate = load_snapshot(metadata_path)
        except Exception:
            continue
        if as_of is not None and candidate.spot_timestamp >= as_of:
            continue
        if threshold is not None and candidate.spot_timestamp > threshold:
            continue
        if baseline == "previous_close" and as_of is not None and candidate.spot_timestamp.date() >= as_of.date():
            continue
        candidate_frame = _prepared_iv_frame(candidate.options_frame(), iv_column)
        if candidate_frame.empty and iv_column != "computedIV":
            candidate_frame = _prepared_iv_frame(candidate.options_frame(), "computedIV")
        if candidate_frame.empty:
            continue
        return candidate, candidate_frame
    return None, pd.DataFrame()


def _matched_change_frame(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    matched = current.merge(
        previous,
        on=["expiration_key", "strike_key", "type_key"],
        suffixes=("_current", "_previous"),
    )
    if matched.empty:
        return matched
    matched["iv_change"] = matched["iv_current"] - matched["iv_previous"]
    matched["iv_change_pct"] = np.where(
        matched["iv_previous"].abs() > 1e-12,
        matched["iv_change"] / matched["iv_previous"],
        np.nan,
    )
    matched["abs_iv_change"] = matched["iv_change"].abs()
    return matched


def _heatmap_records(matched: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (expiry, strike), group in matched.groupby(["expiration_key", "strike_key"], sort=True):
        dte = pd.to_numeric(group["dte_current"], errors="coerce").median()
        change_pct = pd.to_numeric(group["iv_change_pct"], errors="coerce").median()
        rows.append(
            {
                "expiration": expiry,
                "dte": None if pd.isna(dte) else float(dte),
                "strike": float(strike),
                "current_iv": float(group["iv_current"].median()),
                "baseline_iv": float(group["iv_previous"].median()),
                "iv_change": float(group["iv_change"].mean()),
                "iv_change_pct": None if pd.isna(change_pct) else float(change_pct),
                "matched_contracts": int(len(group)),
            }
        )
    return rows


def _tape_record(symbol: str, timestamp: datetime, source: str, mode: str, frame: pd.DataFrame) -> dict[str, Any]:
    grouped = []
    for (expiry, strike), group in frame.groupby(["expiration_key", "strike_key"], sort=True):
        dte = pd.to_numeric(group["dte"], errors="coerce").median()
        grouped.append(
            {
                "expiration": expiry,
                "dte": None if pd.isna(dte) else float(dte),
                "strike": float(strike),
                "iv": float(group["iv"].median()),
                "contracts": int(len(group)),
            }
        )
    return {
        "symbol": symbol.upper(),
        "timestamp": timestamp.isoformat(),
        "source": source,
        "mode": mode,
        "point_count": len(grouped),
        "points": grouped,
    }


def _svi_residual_frame(svi_smiles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for smile in svi_smiles or []:
        expiry = str(smile.get("expiration") or "")
        for item in smile.get("residuals") or []:
            strike = _float_or_none(item.get("strike"))
            observed = _float_or_none(item.get("observed_iv"))
            fitted = _float_or_none(item.get("fitted_iv"))
            if not expiry or strike is None or observed is None or fitted is None:
                continue
            residual = observed - fitted
            rows.append(
                {
                    "expiration_key": expiry,
                    "strike_key": round(strike, 6),
                    "fitted_iv": fitted,
                    "surface_residual": residual,
                    "abs_surface_residual": abs(residual),
                }
            )
    return pd.DataFrame(rows)


def _scanner_chain_frame(chain: pd.DataFrame, iv_column: str) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    column = iv_column if iv_column in chain.columns else "impliedVolatility"
    required = {"expiration", "strike", column}
    if not required.issubset(chain.columns):
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "contract": chain.get("contractSymbol", pd.Series("", index=chain.index)).astype(str),
            "type": chain.get("type", pd.Series("", index=chain.index)).astype(str).str.lower(),
            "expiration": pd.to_datetime(chain["expiration"], errors="coerce"),
            "strike": pd.to_numeric(chain["strike"], errors="coerce"),
            "iv": pd.to_numeric(chain[column], errors="coerce"),
            "dte": pd.to_numeric(chain.get("daysToExpiration"), errors="coerce"),
            "bid_ask_spread_pct": pd.to_numeric(chain.get("bidAskSpreadPct"), errors="coerce"),
            "volume": pd.to_numeric(chain.get("volume"), errors="coerce"),
            "open_interest": pd.to_numeric(chain.get("openInterest"), errors="coerce"),
            "quote_reliability_score": pd.to_numeric(chain.get("quoteReliabilityScore", 1.0), errors="coerce"),
        }
    )
    out = out.dropna(subset=["expiration", "strike", "iv"])
    out = out[(out["strike"] > 0.0) & (out["iv"] > 0.0)]
    if out.empty:
        return out
    out["expiration_key"] = out["expiration"].dt.date.astype(str)
    out["strike_key"] = out["strike"].round(6)
    return out


def _liquidity_scores(frame: pd.DataFrame) -> pd.Series:
    spread = pd.to_numeric(frame.get("bid_ask_spread_pct"), errors="coerce").fillna(0.50).clip(lower=0.0)
    spread_score = (1.0 - (spread / 0.50).clip(upper=1.0)).astype(float)

    volume = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0).clip(lower=0.0)
    open_interest = pd.to_numeric(frame.get("open_interest"), errors="coerce").fillna(0.0).clip(lower=0.0)
    max_volume = float(volume.max()) if len(volume) else 0.0
    max_oi = float(open_interest.max()) if len(open_interest) else 0.0
    volume_score = np.log1p(volume) / np.log1p(max_volume) if max_volume > 0 else pd.Series(0.0, index=frame.index)
    oi_score = np.log1p(open_interest) / np.log1p(max_oi) if max_oi > 0 else pd.Series(0.0, index=frame.index)
    return ((spread_score * 0.50) + (volume_score * 0.25) + (oi_score * 0.25)).clip(0.0, 1.0)


def _scanner_reason(row: pd.Series, direction: str) -> str:
    spread = _finite_or_none(row.get("bid_ask_spread_pct"))
    spread_text = "spread n/a" if spread is None else f"spread {spread:.1%}"
    confidence = _confidence_label(float(row.get("candidate_confidence") or 0.0))
    return (
        f"{direction.title()} to fitted SVI by {row['surface_residual']:.2%}; "
        f"z-score {row['residual_z_score']:.2f}; "
        f"{confidence} confidence; {spread_text}; OI {int(row.get('open_interest') or 0)}; "
        f"volume {int(row.get('volume') or 0)}"
    )


def _confidence_label(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def _scanner_unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "source": "current_chain_plus_svi_fit",
        "model": "SVI",
        "fit_mode": "Robust SVI",
        "ranking_policy": "abs_residual_z_score_times_liquidity_and_quote_reliability",
        "reason": reason,
        "candidate_count": 0,
        "rich_count": 0,
        "cheap_count": 0,
        "candidates": [],
    }


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
        "heatmaps": {"available": False, "reason": reason, "baselines": {}},
        "tape": {"available": False, "reason": reason, "snapshots": []},
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


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _finite_or_none(value: Any) -> float | None:
    return _float_or_none(value)
