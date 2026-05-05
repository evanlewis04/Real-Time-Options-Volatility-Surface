"""Surface-grid helpers used by the dashboard views."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.pricing.black_scholes import OptionGreeks
from src.quant.skew import term_structure_metrics


def surface_mesh(strikes, expiries, surface):
    strike_arr = np.asarray(strikes)
    expiry_arr = np.asarray(expiries)
    if strike_arr.ndim == 1 and expiry_arr.ndim == 1:
        strike_mesh, expiry_mesh = np.meshgrid(strike_arr, expiry_arr)
    else:
        strike_mesh, expiry_mesh = strike_arr, expiry_arr
    return strike_mesh, expiry_mesh, np.asarray(surface)


def surface_axis(
    strikes,
    expiries,
    surface,
    spot: float,
    axis: str = "strike",
    risk_free_rate: float | None = None,
    dividend_yield: float | None = None,
):
    """Return the selected surface x-axis mesh plus chart labels and hover formatting."""
    strike_mesh, expiry_mesh, vols = surface_mesh(strikes, expiries, surface)
    axis_key = str(axis or "strike").lower().replace(" ", "_").replace("-", "_")

    if axis_key in {"moneyness", "spot_moneyness"}:
        values = strike_mesh / spot
        return values, expiry_mesh, vols, "Moneyness (K/S)", "%{x:.3f}", "Moneyness: %{x:.3f}"

    if axis_key in {"log_moneyness", "logmoneyness"}:
        values = np.log(np.where(strike_mesh > 0, strike_mesh / spot, np.nan))
        return values, expiry_mesh, vols, "Log-moneyness ln(K/S)", "%{x:.3f}", "Log-moneyness: %{x:.3f}"

    if axis_key in {"delta", "call_delta"}:
        values = _call_delta_axis(strike_mesh, expiry_mesh, vols, spot, risk_free_rate, dividend_yield)
        return values, expiry_mesh, vols, "Call delta", "%{x:.3f}", "Call delta: %{x:.3f}"

    return strike_mesh, expiry_mesh, vols, "Strike", "$%{x:.2f}", "Strike: $%{x:.2f}"


def extract_smile(strikes, expiries, surface, spot: float | None = None, axis: str = "strike"):
    strike_mesh, expiry_mesh, vol_surface = surface_mesh(strikes, expiries, surface)
    row = 0
    smile_strikes = strike_mesh[row, :] if strike_mesh.ndim == 2 else strike_mesh
    smile_vols = vol_surface[row, :] if vol_surface.ndim == 2 else vol_surface
    smile_days = expiry_mesh[row, 0] if expiry_mesh.ndim == 2 else expiry_mesh[row]
    if spot is None:
        return smile_strikes, smile_vols, float(smile_days)
    axis_mesh, _, _, label, _, hover_label = surface_axis(strikes, expiries, surface, spot, axis)
    smile_axis = axis_mesh[row, :] if axis_mesh.ndim == 2 else axis_mesh
    return smile_axis, smile_vols, float(smile_days), label, hover_label


def surface_stats(strikes, expiries, surface, spot: float) -> Dict[str, Any]:
    strike_mesh, expiry_mesh, vols = surface_mesh(strikes, expiries, surface)
    finite = vols[np.isfinite(vols)]
    if finite.size == 0:
        return {}
    atm_idx = np.unravel_index(np.nanargmin(np.abs(strike_mesh - spot)), strike_mesh.shape)
    expiries_1d = expiry_mesh[:, 0] if expiry_mesh.ndim == 2 else expiry_mesh
    strike_1d = strike_mesh[0, :] if strike_mesh.ndim == 2 else strike_mesh
    atm_term = []
    for i, dte in enumerate(expiries_1d):
        row_strikes = strike_mesh[i, :] if strike_mesh.ndim == 2 else strike_1d
        idx = int(np.nanargmin(np.abs(row_strikes - spot)))
        atm_term.append((float(dte), float(vols[i, idx])))
    front = atm_term[0][1] if atm_term else None
    back = atm_term[-1][1] if atm_term else None
    return {
        "atm_iv": float(vols[atm_idx]),
        "min_iv": float(np.nanmin(vols)),
        "max_iv": float(np.nanmax(vols)),
        "term_spread": None if front is None or back is None else back - front,
        "term_metrics": term_structure_metrics(atm_term),
        "points": int(np.size(vols)),
        "atm_term": atm_term,
        "strike_min": float(np.nanmin(strike_1d)),
        "strike_max": float(np.nanmax(strike_1d)),
    }


def _call_delta_axis(
    strike_mesh: np.ndarray,
    expiry_mesh: np.ndarray,
    vols: np.ndarray,
    spot: float,
    risk_free_rate: float | None,
    dividend_yield: float | None,
) -> np.ndarray:
    rate = float(risk_free_rate or 0.0)
    dividend = float(dividend_yield or 0.0)
    out = np.full_like(strike_mesh, np.nan, dtype=float)
    for index in np.ndindex(strike_mesh.shape):
        strike = float(strike_mesh[index])
        dte = float(expiry_mesh[index])
        vol = float(vols[index])
        if spot <= 0 or strike <= 0 or dte <= 0 or not np.isfinite(vol) or vol <= 0:
            continue
        out[index] = OptionGreeks.delta(spot, strike, dte / 365.0, rate, vol, "call", dividend)
    return out
