"""Historical volatility-surface prior loader from persisted snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.models import MarketDataSnapshot
from src.data.snapshots import load_recent_snapshots
from src.quant.provenance import CURRENT_ROBUST_FIT_PROVENANCE, HISTORICAL_PRIOR_PROVENANCE


@dataclass(frozen=True)
class HistoricalSurfacePrior:
    """Prior grid plus provenance for later prior-assisted fitting."""

    available: bool
    symbol: str
    source: str = "persisted_snapshots"
    reason: str | None = None
    grid: pd.DataFrame = field(default_factory=pd.DataFrame)
    snapshot_timestamps: tuple[str, ...] = ()
    latest_snapshot_timestamp: str | None = None
    latest_age_days: float | None = None
    snapshot_count: int = 0
    source_point_count: int = 0
    cell_count: int = 0
    min_snapshots: int = 2
    min_points: int = 12
    max_age_days: float | None = None

    def metadata(self) -> dict[str, Any]:
        """Return dashboard-safe metadata without the DataFrame payload."""
        return {
            "available": self.available,
            "source": self.source,
            "reason": self.reason,
            "symbol": self.symbol,
            "snapshot_count": self.snapshot_count,
            "source_point_count": self.source_point_count,
            "cell_count": self.cell_count,
            "latest_snapshot_timestamp": self.latest_snapshot_timestamp,
            "latest_age_days": self.latest_age_days,
            "snapshot_timestamps": list(self.snapshot_timestamps),
            "min_snapshots": self.min_snapshots,
            "min_points": self.min_points,
            "max_age_days": self.max_age_days,
            "provenance": HISTORICAL_PRIOR_PROVENANCE,
        }

    def records(self) -> list[dict[str, Any]]:
        """Return deterministic row records for serialization/tests."""
        if self.grid.empty:
            return []
        return self.grid.replace({np.nan: None}).to_dict("records")


def load_historical_surface_prior(
    symbol: str,
    directory: Path | str,
    *,
    as_of: datetime | pd.Timestamp | None = None,
    iv_column: str = "computedIV",
    max_age: timedelta = timedelta(days=5),
    max_snapshots: int = 12,
    min_snapshots: int = 2,
    min_points: int = 12,
    log_moneyness_step: float = 0.05,
    dte_step: int = 7,
) -> HistoricalSurfacePrior:
    """Build a deterministic historical prior grid from recent snapshots."""
    symbol_key = symbol.upper()
    as_of_dt = _timestamp_or_none(as_of) or datetime.now()
    snapshots = load_recent_snapshots(symbol_key, directory, before=as_of_dt, max_count=max_snapshots)
    if not snapshots:
        return _unavailable(symbol_key, "No earlier persisted snapshots", min_snapshots, min_points, max_age)

    latest_age = as_of_dt - snapshots[0].spot_timestamp
    if latest_age > max_age:
        return _unavailable(
            symbol_key,
            "Latest persisted snapshot is stale",
            min_snapshots,
            min_points,
            max_age,
            snapshots=snapshots,
            latest_age=latest_age,
        )

    usable: list[tuple[MarketDataSnapshot, pd.DataFrame]] = []
    for snapshot in snapshots:
        age = as_of_dt - snapshot.spot_timestamp
        if age < timedelta(0) or age > max_age:
            continue
        frame = _prepared_prior_frame(
            snapshot.options_frame(),
            snapshot.spot,
            iv_column,
            log_moneyness_step=log_moneyness_step,
            dte_step=dte_step,
        )
        if not frame.empty:
            usable.append((snapshot, frame))

    if len(usable) < min_snapshots:
        return _unavailable(
            symbol_key,
            f"Need at least {min_snapshots} recent snapshots with usable IV rows",
            min_snapshots,
            min_points,
            max_age,
            snapshots=[snapshot for snapshot, _ in usable] or snapshots,
            latest_age=latest_age,
        )

    source = pd.concat(
        [
            frame.assign(
                snapshot_timestamp=snapshot.spot_timestamp.isoformat(),
                snapshot_source=snapshot.source,
                snapshot_mode=snapshot.mode,
            )
            for snapshot, frame in usable
        ],
        ignore_index=True,
    )
    if len(source) < min_points:
        return _unavailable(
            symbol_key,
            f"Need at least {min_points} historical IV points",
            min_snapshots,
            min_points,
            max_age,
            snapshots=[snapshot for snapshot, _ in usable],
            latest_age=latest_age,
        )

    grid = _prior_grid(source)
    if grid.empty:
        return _unavailable(
            symbol_key,
            "No prior grid cells could be constructed",
            min_snapshots,
            min_points,
            max_age,
            snapshots=[snapshot for snapshot, _ in usable],
            latest_age=latest_age,
        )

    timestamps = tuple(snapshot.spot_timestamp.isoformat() for snapshot, _ in usable)
    return HistoricalSurfacePrior(
        available=True,
        symbol=symbol_key,
        reason=None,
        grid=grid,
        snapshot_timestamps=timestamps,
        latest_snapshot_timestamp=timestamps[0],
        latest_age_days=latest_age.total_seconds() / 86400.0,
        snapshot_count=len(usable),
        source_point_count=int(len(source)),
        cell_count=int(len(grid)),
        min_snapshots=min_snapshots,
        min_points=min_points,
        max_age_days=max_age.total_seconds() / 86400.0,
    )


def blend_surface_with_prior(
    strikes: Any,
    expiries: Any,
    vols: Any,
    spot: float,
    prior: HistoricalSurfacePrior,
    *,
    quality_score: float | None,
    min_quality_score: float = 70.0,
    max_blend_weight: float = 0.35,
    jump_threshold: float = 0.04,
    jump_directional_share: float = 0.75,
    jump_changed_share: float = 0.60,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend a current fitted surface with a historical prior when quality is poor."""
    current = np.asarray(vols, dtype=float).copy()
    metadata = _blend_metadata(prior, applied=False, reason="Prior unavailable")
    if not prior.available or prior.grid.empty:
        metadata["reason"] = prior.reason or "Prior unavailable"
        return current, metadata
    if quality_score is None or not np.isfinite(float(quality_score)):
        metadata["reason"] = "Current quality score unavailable"
        return current, metadata
    if float(quality_score) >= min_quality_score:
        metadata["reason"] = "Current quality score is adequate"
        return current, metadata
    if current.size == 0 or spot <= 0.0:
        metadata["reason"] = "Current surface grid unavailable"
        return current, metadata

    strike_grid, expiry_grid = _surface_mesh(strikes, expiries, current.shape)
    prior_values, overlap = _nearest_prior_values(strike_grid, expiry_grid, spot, prior.grid)
    overlap_count = int(np.count_nonzero(overlap & np.isfinite(current)))
    metadata["overlap_count"] = overlap_count
    if overlap_count == 0:
        metadata["reason"] = "No current surface cells overlap the prior grid"
        return current, metadata
    jump = _surface_jump_detection(
        current,
        prior_values,
        overlap,
        threshold=jump_threshold,
        directional_share_threshold=jump_directional_share,
        changed_share_threshold=jump_changed_share,
    )
    metadata["jump_detection"] = jump
    if jump["broad_shift_detected"]:
        metadata["reason"] = "Current clean quotes indicate broad IV shift"
        return current, metadata

    total_cells = max(int(np.count_nonzero(np.isfinite(current))), 1)
    quality_factor = np.clip((min_quality_score - float(quality_score)) / min_quality_score, 0.0, 1.0)
    recency_factor = _recency_factor(prior.latest_age_days, prior.max_age_days)
    overlap_factor = np.clip(overlap_count / total_cells, 0.0, 1.0)
    weight = float(np.clip(max_blend_weight * quality_factor * recency_factor * overlap_factor, 0.0, max_blend_weight))
    metadata.update(
        {
            "quality_score": float(quality_score),
            "min_quality_score": float(min_quality_score),
            "recency_factor": recency_factor,
            "overlap_factor": float(overlap_factor),
            "blend_weight": weight,
        }
    )
    if weight <= 0.0:
        metadata["reason"] = "Computed prior blend weight is zero"
        return current, metadata

    blended = current.copy()
    blend_mask = overlap & np.isfinite(current) & np.isfinite(prior_values)
    blended[blend_mask] = ((1.0 - weight) * current[blend_mask]) + (weight * prior_values[blend_mask])
    metadata.update(
        {
            "applied": True,
            "reason": None,
            "policy": "quality_recency_overlap_weighted_prior_assist",
            "blend_weight": weight,
            "blended_cell_count": int(np.count_nonzero(blend_mask)),
        }
    )
    return blended, metadata


def surface_prior_comparison_records(
    strikes: Any,
    expiries: Any,
    vols: Any,
    spot: float,
    prior: HistoricalSurfacePrior,
) -> list[dict[str, Any]]:
    """Return current fit, prior estimate, and current-minus-prior rows."""
    current = np.asarray(vols, dtype=float)
    if not prior.available or prior.grid.empty or current.size == 0 or spot <= 0.0:
        return []
    strike_grid, expiry_grid = _surface_mesh(strikes, expiries, current.shape)
    prior_values, overlap = _nearest_prior_values(strike_grid, expiry_grid, spot, prior.grid)
    mask = overlap & np.isfinite(current) & np.isfinite(prior_values)
    if not np.any(mask):
        return []

    log_moneyness = np.log(np.where(strike_grid > 0.0, strike_grid / float(spot), np.nan))
    rows = []
    for index in np.argwhere(mask):
        idx = tuple(index)
        current_iv = float(current[idx])
        prior_iv = float(prior_values[idx])
        rows.append(
            {
                "strike": float(strike_grid[idx]),
                "dte": float(expiry_grid[idx]),
                "log_moneyness": float(log_moneyness[idx]),
                "current_iv": current_iv,
                "prior_iv": prior_iv,
                "iv_change": current_iv - prior_iv,
                "abs_iv_change": abs(current_iv - prior_iv),
                "prior_timestamp": prior.latest_snapshot_timestamp,
                "prior_age_days": prior.latest_age_days,
                "source": prior.source,
                "provenance": HISTORICAL_PRIOR_PROVENANCE,
                "current_label": CURRENT_ROBUST_FIT_PROVENANCE,
                "prior_label": HISTORICAL_PRIOR_PROVENANCE,
            }
        )
    return sorted(rows, key=lambda row: (row["dte"], row["strike"]))


def _prepared_prior_frame(
    chain: pd.DataFrame,
    spot: float,
    iv_column: str,
    *,
    log_moneyness_step: float,
    dte_step: int,
) -> pd.DataFrame:
    if chain.empty or spot <= 0.0:
        return pd.DataFrame()
    column = iv_column if iv_column in chain.columns else "impliedVolatility"
    required = {"expiration", "strike", "daysToExpiration", column}
    if not required.issubset(chain.columns):
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "expiration": pd.to_datetime(chain["expiration"], errors="coerce"),
            "strike": pd.to_numeric(chain["strike"], errors="coerce"),
            "dte": pd.to_numeric(chain["daysToExpiration"], errors="coerce"),
            "iv": pd.to_numeric(chain[column], errors="coerce"),
            "log_moneyness": _log_moneyness(chain, spot),
        }
    )
    out = out.dropna(subset=["expiration", "strike", "dte", "iv", "log_moneyness"])
    out = out[(out["strike"] > 0.0) & (out["dte"] > 0.0) & (out["iv"] > 0.0)]
    if out.empty:
        return out
    out["dte_bucket"] = _round_to_step(out["dte"], float(dte_step)).astype(float)
    out["log_moneyness_bucket"] = _round_to_step(out["log_moneyness"], float(log_moneyness_step))
    out["moneyness_bucket"] = np.exp(out["log_moneyness_bucket"])
    out["expiration_key"] = out["expiration"].dt.date.astype(str)
    return out.sort_values(["dte_bucket", "log_moneyness_bucket", "expiration_key"]).reset_index(drop=True)


def _prior_grid(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = source.groupby(["dte_bucket", "log_moneyness_bucket"], sort=True)
    for (dte_bucket, log_money_bucket), group in grouped:
        rows.append(
            {
                "dte": float(dte_bucket),
                "log_moneyness": float(log_money_bucket),
                "moneyness": float(np.exp(log_money_bucket)),
                "prior_iv": float(group["iv"].median()),
                "prior_iv_mean": float(group["iv"].mean()),
                "prior_iv_std": _std_or_none(group["iv"]),
                "observations": int(len(group)),
                "snapshot_count": int(group["snapshot_timestamp"].nunique()),
                "source": "persisted_snapshots",
                "provenance": HISTORICAL_PRIOR_PROVENANCE,
                "snapshot_timestamps": sorted(group["snapshot_timestamp"].unique().tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dte", "log_moneyness"]).reset_index(drop=True)


def _log_moneyness(chain: pd.DataFrame, spot: float) -> pd.Series:
    if "logMoneyness" in chain:
        values = pd.to_numeric(chain["logMoneyness"], errors="coerce")
        if values.notna().any():
            return values
    strikes = pd.to_numeric(chain["strike"], errors="coerce")
    if "forwardPrice" in chain:
        forwards = pd.to_numeric(chain["forwardPrice"], errors="coerce")
        return np.log(strikes / forwards.where(forwards > 0.0))
    return np.log(strikes / float(spot))


def _round_to_step(values: pd.Series, step: float) -> pd.Series:
    safe_step = max(float(step), 1e-12)
    return (pd.to_numeric(values, errors="coerce") / safe_step).round() * safe_step


def _std_or_none(values: pd.Series) -> float | None:
    std = float(pd.to_numeric(values, errors="coerce").std(ddof=0))
    return std if np.isfinite(std) else None


def _timestamp_or_none(value: datetime | pd.Timestamp | None) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _surface_mesh(strikes: Any, expiries: Any, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    strike_values = np.asarray(strikes, dtype=float)
    expiry_values = np.asarray(expiries, dtype=float)
    if strike_values.shape == shape and expiry_values.shape == shape:
        return strike_values, expiry_values
    if len(shape) == 2 and strike_values.ndim == 1 and expiry_values.ndim == 1:
        return np.meshgrid(strike_values, expiry_values)
    if strike_values.shape == shape and expiry_values.ndim == 1 and len(shape) == 2:
        return strike_values, np.repeat(expiry_values.reshape(-1, 1), shape[1], axis=1)
    if expiry_values.shape == shape and strike_values.ndim == 1 and len(shape) == 2:
        return np.repeat(strike_values.reshape(1, -1), shape[0], axis=0), expiry_values
    return np.broadcast_to(strike_values, shape), np.broadcast_to(expiry_values, shape)


def _nearest_prior_values(
    strike_grid: np.ndarray,
    expiry_grid: np.ndarray,
    spot: float,
    prior_grid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    prior = prior_grid.dropna(subset=["dte", "log_moneyness", "prior_iv"])
    if prior.empty:
        return np.full(strike_grid.shape, np.nan), np.zeros(strike_grid.shape, dtype=bool)

    current_log_money = np.log(np.where(strike_grid > 0.0, strike_grid / float(spot), np.nan))
    dte_values = prior["dte"].to_numpy(dtype=float)
    log_money_values = prior["log_moneyness"].to_numpy(dtype=float)
    iv_values = prior["prior_iv"].to_numpy(dtype=float)
    dte_scale = max(float(np.nanmax(dte_values) - np.nanmin(dte_values)), 1.0)
    log_money_scale = max(float(np.nanmax(log_money_values) - np.nanmin(log_money_values)), 0.05)

    dte_tolerance = 1e-9
    log_money_tolerance = 1e-9
    overlap = (
        np.isfinite(current_log_money)
        & np.isfinite(expiry_grid)
        & (expiry_grid >= np.nanmin(dte_values) - dte_tolerance)
        & (expiry_grid <= np.nanmax(dte_values) + dte_tolerance)
        & (current_log_money >= np.nanmin(log_money_values) - log_money_tolerance)
        & (current_log_money <= np.nanmax(log_money_values) + log_money_tolerance)
    )
    out = np.full(strike_grid.shape, np.nan)
    if not np.any(overlap):
        return out, overlap

    current_points = np.column_stack((expiry_grid[overlap], current_log_money[overlap]))
    prior_points = np.column_stack((dte_values, log_money_values))
    distances = (
        ((current_points[:, None, 0] - prior_points[None, :, 0]) / dte_scale) ** 2
        + ((current_points[:, None, 1] - prior_points[None, :, 1]) / log_money_scale) ** 2
    )
    nearest = np.argmin(distances, axis=1)
    out[overlap] = iv_values[nearest]
    return out, overlap


def _recency_factor(latest_age_days: float | None, max_age_days: float | None) -> float:
    if latest_age_days is None or max_age_days is None or max_age_days <= 0.0:
        return 0.0
    return float(np.clip(1.0 - (latest_age_days / max_age_days), 0.0, 1.0))


def _surface_jump_detection(
    current: np.ndarray,
    prior_values: np.ndarray,
    overlap: np.ndarray,
    *,
    threshold: float,
    directional_share_threshold: float,
    changed_share_threshold: float,
) -> dict[str, Any]:
    mask = overlap & np.isfinite(current) & np.isfinite(prior_values)
    if not np.any(mask):
        return {
            "broad_shift_detected": False,
            "overlap_count": 0,
            "median_change": None,
            "median_abs_change": None,
            "directional_share": None,
            "changed_share": None,
            "threshold": float(threshold),
        }
    changes = current[mask] - prior_values[mask]
    abs_changes = np.abs(changes)
    up_share = float(np.mean(changes > 0.0))
    down_share = float(np.mean(changes < 0.0))
    directional_share = max(up_share, down_share)
    changed_share = float(np.mean(abs_changes >= threshold))
    median_change = float(np.median(changes))
    median_abs_change = float(np.median(abs_changes))
    broad = (
        abs(median_change) >= threshold
        and directional_share >= directional_share_threshold
        and changed_share >= changed_share_threshold
    )
    return {
        "broad_shift_detected": bool(broad),
        "overlap_count": int(len(changes)),
        "median_change": median_change,
        "median_abs_change": median_abs_change,
        "directional_share": directional_share,
        "changed_share": changed_share,
        "threshold": float(threshold),
        "directional_share_threshold": float(directional_share_threshold),
        "changed_share_threshold": float(changed_share_threshold),
    }


def _blend_metadata(prior: HistoricalSurfacePrior, *, applied: bool, reason: str | None) -> dict[str, Any]:
    return {
        "available": prior.available,
        "applied": applied,
        "reason": reason,
        "source": prior.source,
        "prior_source": prior.source,
        "prior_age_days": prior.latest_age_days,
        "prior_timestamp": prior.latest_snapshot_timestamp,
        "prior_snapshot_count": prior.snapshot_count,
        "prior_cell_count": prior.cell_count,
        "overlap_count": 0,
        "blend_weight": 0.0,
        "blended_cell_count": 0,
        "policy": "quality_recency_overlap_weighted_prior_assist",
        "provenance": HISTORICAL_PRIOR_PROVENANCE,
    }


def _unavailable(
    symbol: str,
    reason: str,
    min_snapshots: int,
    min_points: int,
    max_age: timedelta,
    *,
    snapshots: list[MarketDataSnapshot] | None = None,
    latest_age: timedelta | None = None,
) -> HistoricalSurfacePrior:
    timestamps = tuple(snapshot.spot_timestamp.isoformat() for snapshot in snapshots or [])
    return HistoricalSurfacePrior(
        available=False,
        symbol=symbol,
        reason=reason,
        snapshot_timestamps=timestamps,
        latest_snapshot_timestamp=timestamps[0] if timestamps else None,
        latest_age_days=latest_age.total_seconds() / 86400.0 if latest_age is not None else None,
        snapshot_count=len(timestamps),
        min_snapshots=min_snapshots,
        min_points=min_points,
        max_age_days=max_age.total_seconds() / 86400.0,
    )
