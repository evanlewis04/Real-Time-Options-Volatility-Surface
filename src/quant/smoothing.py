"""Arbitrage-aware surface smoothing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def smooth_iv_surface(
    strikes: Any,
    expiries: Any,
    vols: Any,
    *,
    sigma: float = 0.65,
    blend: float = 0.70,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Smooth an IV grid and enforce nondecreasing total variance by expiry."""
    vol_grid = np.asarray(vols, dtype=float)
    if vol_grid.size == 0:
        return vol_grid, _metadata(False, 0.0, 0.0, 0, 0)

    expiry_grid = _expiry_mesh(expiries, vol_grid.shape)
    rough_before = _roughness(vol_grid)
    try:
        from scipy.ndimage import gaussian_filter

        smoothed = gaussian_filter(vol_grid, sigma=float(sigma), mode="nearest")
        smoothed = blend * smoothed + (1.0 - blend) * vol_grid
        applied = True
    except ImportError:
        smoothed = vol_grid.copy()
        applied = False

    smoothed = np.clip(smoothed, 0.01, 5.0)
    calendar_adjustments = _enforce_calendar_total_variance(smoothed, expiry_grid)
    rough_after = _roughness(smoothed)
    convexity_penalty = _convexity_penalty(smoothed)
    return smoothed, _metadata(applied, rough_before, rough_after, calendar_adjustments, convexity_penalty)


def smoothing_summary(strikes: Any, expiries: Any, vols: Any) -> dict[str, Any]:
    """Return lightweight diagnostics for an already fitted surface."""
    vol_grid = np.asarray(vols, dtype=float)
    return {
        "method": "gaussian_blend_calendar_total_variance",
        "surface_shape": list(vol_grid.shape),
        "roughness": _roughness(vol_grid),
        "convexity_penalty": _convexity_penalty(vol_grid),
        "strike_points": int(np.asarray(strikes).shape[-1]) if np.asarray(strikes).size else 0,
        "expiry_points": int(np.asarray(expiries).shape[0]) if np.asarray(expiries).size else 0,
    }


def _metadata(
    applied: bool,
    rough_before: float,
    rough_after: float,
    calendar_adjustments: int,
    convexity_penalty: int,
) -> dict[str, Any]:
    return {
        "method": "gaussian_blend_calendar_total_variance",
        "applied": bool(applied),
        "roughness_before": float(rough_before),
        "roughness_after": float(rough_after),
        "roughness_reduction": float(max(0.0, rough_before - rough_after)),
        "calendar_adjustments": int(calendar_adjustments),
        "convexity_penalty": int(convexity_penalty),
    }


def _expiry_mesh(expiries: Any, shape: tuple[int, ...]) -> np.ndarray:
    expiry = np.asarray(expiries, dtype=float)
    if expiry.shape == shape:
        return expiry
    if expiry.ndim == 1 and len(shape) == 2:
        return np.repeat(expiry.reshape(-1, 1), shape[1], axis=1)
    return np.ones(shape, dtype=float)


def _enforce_calendar_total_variance(vol_grid: np.ndarray, expiry_grid: np.ndarray) -> int:
    if vol_grid.ndim != 2 or expiry_grid.shape != vol_grid.shape:
        return 0
    years = np.maximum(expiry_grid / 365.0, 1e-8)
    total_variance = vol_grid**2 * years
    adjusted = np.maximum.accumulate(total_variance, axis=0)
    adjustments = int(np.sum(adjusted > total_variance + 1e-10))
    vol_grid[:] = np.sqrt(np.maximum(adjusted / years, 1e-10))
    return adjustments


def _roughness(vol_grid: np.ndarray) -> float:
    finite = np.where(np.isfinite(vol_grid), vol_grid, np.nan)
    if finite.size == 0:
        return 0.0
    parts = []
    if finite.shape[0] > 1:
        parts.append(np.nanmean(np.diff(finite, axis=0) ** 2))
    if finite.ndim > 1 and finite.shape[1] > 1:
        parts.append(np.nanmean(np.diff(finite, axis=1) ** 2))
    clean = [float(item) for item in parts if np.isfinite(item)]
    return float(sum(clean)) if clean else 0.0


def _convexity_penalty(vol_grid: np.ndarray) -> int:
    if vol_grid.ndim != 2 or vol_grid.shape[1] < 3:
        return 0
    second_diff = np.diff(vol_grid, n=2, axis=1)
    return int(np.sum(second_diff < -1e-6))
