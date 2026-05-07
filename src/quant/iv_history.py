"""Persisted-snapshot implied-volatility history analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.snapshots import list_snapshots, load_snapshot


def atm_iv_from_chain(chain: pd.DataFrame, spot: float, target_dte: int = 30, dte_tolerance: int = 14) -> float | None:
    """Return a representative near-ATM IV for an option chain."""
    if chain.empty or spot <= 0:
        return None
    iv_column = "computedIV" if "computedIV" in chain else "impliedVolatility"
    required = {"strike", "daysToExpiration", iv_column}
    if not required.issubset(chain.columns):
        return None

    work = chain.copy()
    work["iv_num"] = pd.to_numeric(work[iv_column], errors="coerce")
    work["strike_num"] = pd.to_numeric(work["strike"], errors="coerce")
    work["dte_num"] = pd.to_numeric(work["daysToExpiration"], errors="coerce")
    work = work.dropna(subset=["iv_num", "strike_num", "dte_num"])
    work = work[(work["iv_num"] > 0.0) & (work["dte_num"] > 0.0)]
    if work.empty:
        return None
    near = work[np.abs(work["dte_num"] - float(target_dte)) <= float(dte_tolerance)].copy()
    if near.empty:
        nearest_dte = float((work["dte_num"] - float(target_dte)).abs().min())
        near = work[(work["dte_num"] - float(target_dte)).abs() <= nearest_dte].copy()
    near["atm_distance"] = (near["strike_num"] - float(spot)).abs()
    sample = near.sort_values(["atm_distance", "dte_num"]).head(6)
    ivs = sample["iv_num"].dropna()
    return float(ivs.median()) if not ivs.empty else None


def iv_rank_percentile_from_snapshots(
    symbol: str,
    current_iv: float | None,
    directory: Path | str,
    *,
    min_points: int = 3,
) -> dict[str, Any]:
    """Compute IV rank and percentile from persisted local snapshots."""
    if current_iv is None or not np.isfinite(current_iv) or current_iv <= 0.0:
        return _unavailable("Current ATM IV is unavailable")

    observations: list[dict[str, Any]] = []
    for metadata_path in list_snapshots(symbol, directory):
        try:
            snapshot = load_snapshot(metadata_path)
        except Exception:
            continue
        iv = atm_iv_from_chain(snapshot.options_frame(), snapshot.spot)
        if iv is None:
            continue
        observations.append(
            {
                "timestamp": snapshot.spot_timestamp.isoformat(),
                "atm_iv": iv,
                "source": snapshot.source,
                "mode": snapshot.mode,
            }
        )

    if len(observations) < min_points:
        return _unavailable(f"Need at least {min_points} stored snapshots with ATM IV", observations)

    history = np.array([item["atm_iv"] for item in observations], dtype=float)
    min_iv = float(np.nanmin(history))
    max_iv = float(np.nanmax(history))
    iv_range = max_iv - min_iv
    rank = None if iv_range <= 1e-12 else float((float(current_iv) - min_iv) / iv_range)
    percentile = float(np.mean(history <= float(current_iv)))
    return {
        "available": True,
        "source": "persisted_snapshots",
        "current_iv": float(current_iv),
        "observations": int(len(history)),
        "min_iv": min_iv,
        "max_iv": max_iv,
        "iv_rank": None if rank is None else float(np.clip(rank, 0.0, 1.0)),
        "iv_percentile": float(np.clip(percentile, 0.0, 1.0)),
        "history": observations[:120],
    }


def _unavailable(reason: str, observations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "source": "persisted_snapshots",
        "reason": reason,
        "observations": len(observations or []),
        "iv_rank": None,
        "iv_percentile": None,
        "history": observations or [],
    }
