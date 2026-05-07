"""Dupire-style local-volatility approximation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


LOCAL_VOL_WARNINGS = [
    "Dupire local volatility is an approximation and is highly sensitive to surface smoothing.",
    "Use only when quote quality is strong and calendar-smoothed total variance is available.",
]


def dupire_local_vol_surface(
    strikes: Any,
    expiries: Any,
    implied_vols: Any,
    spot: float,
    *,
    quality_score: float | None = None,
    smoothing_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Approximate local volatility from a smoothed implied-volatility grid."""
    strike_grid, expiry_grid, vol_grid = _mesh_inputs(strikes, expiries, implied_vols)
    gate = _gate_local_vol(strike_grid, expiry_grid, vol_grid, spot, quality_score, smoothing_meta)
    if not gate["enabled"]:
        return {
            "enabled": False,
            "reason": gate["reason"],
            "warnings": LOCAL_VOL_WARNINGS,
            "grid": [],
            "invalid_points": None,
            "min_local_vol": None,
            "max_local_vol": None,
        }

    years = np.maximum(expiry_grid / 365.0, 1e-8)
    total_variance = np.maximum(vol_grid, 1e-6) ** 2 * years
    log_moneyness = np.log(np.maximum(strike_grid, 1e-8) / float(spot))
    y_axis = log_moneyness[0, :] if log_moneyness.ndim == 2 else log_moneyness
    t_axis = years[:, 0] if years.ndim == 2 else years

    dw_dt = np.gradient(total_variance, t_axis, axis=0, edge_order=1)
    dw_dy = np.gradient(total_variance, y_axis, axis=1, edge_order=1)
    d2w_dy2 = np.gradient(dw_dy, y_axis, axis=1, edge_order=1)
    w = np.maximum(total_variance, 1e-8)
    y = log_moneyness
    denominator = (
        1.0
        - (y / w) * dw_dy
        + 0.25 * (-0.25 - 1.0 / w + (y**2) / (w**2)) * (dw_dy**2)
        + 0.5 * d2w_dy2
    )
    local_var = np.where(denominator > 1e-8, dw_dt / denominator, np.nan)
    local_vol = np.sqrt(np.where(local_var > 0.0, local_var, np.nan))
    invalid = int(np.size(local_vol) - np.isfinite(local_vol).sum())
    clean = np.clip(local_vol, 0.01, 5.0)
    finite = clean[np.isfinite(clean)]
    return {
        "enabled": bool(finite.size),
        "reason": None if finite.size else "Dupire denominator produced no positive finite local variances",
        "warnings": LOCAL_VOL_WARNINGS,
        "grid": np.where(np.isfinite(clean), clean, np.nan).tolist(),
        "invalid_points": invalid,
        "min_local_vol": float(np.nanmin(clean)) if finite.size else None,
        "max_local_vol": float(np.nanmax(clean)) if finite.size else None,
        "method": "dupire_total_variance_log_moneyness",
    }


def _gate_local_vol(
    strike_grid: np.ndarray,
    expiry_grid: np.ndarray,
    vol_grid: np.ndarray,
    spot: float,
    quality_score: float | None,
    smoothing_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    if spot <= 0:
        return {"enabled": False, "reason": "Spot must be positive"}
    if vol_grid.ndim != 2 or min(vol_grid.shape) < 3:
        return {"enabled": False, "reason": "Local vol requires at least a 3x3 IV grid"}
    if strike_grid.shape != vol_grid.shape or expiry_grid.shape != vol_grid.shape:
        return {"enabled": False, "reason": "Strike, expiry, and IV grids must have matching shapes"}
    if not np.isfinite(vol_grid).all():
        return {"enabled": False, "reason": "IV grid contains non-finite values"}
    if quality_score is None or quality_score < 70.0:
        return {"enabled": False, "reason": "Surface quality score must be at least 70"}
    if not smoothing_meta or not smoothing_meta.get("method"):
        return {"enabled": False, "reason": "Smoothed total-variance surface metadata is required"}
    return {"enabled": True, "reason": None}


def _mesh_inputs(strikes: Any, expiries: Any, implied_vols: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vol_grid = np.asarray(implied_vols, dtype=float)
    strike_arr = np.asarray(strikes, dtype=float)
    expiry_arr = np.asarray(expiries, dtype=float)
    if strike_arr.shape != vol_grid.shape:
        strike_arr = np.repeat(strike_arr.reshape(1, -1), vol_grid.shape[0], axis=0)
    if expiry_arr.shape != vol_grid.shape:
        expiry_arr = np.repeat(expiry_arr.reshape(-1, 1), vol_grid.shape[1], axis=1)
    return strike_arr, expiry_arr, vol_grid
