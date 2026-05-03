"""Surface-grid helpers used by the dashboard views."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def surface_mesh(strikes, expiries, surface):
    strike_arr = np.asarray(strikes)
    expiry_arr = np.asarray(expiries)
    if strike_arr.ndim == 1 and expiry_arr.ndim == 1:
        strike_mesh, expiry_mesh = np.meshgrid(strike_arr, expiry_arr)
    else:
        strike_mesh, expiry_mesh = strike_arr, expiry_arr
    return strike_mesh, expiry_mesh, np.asarray(surface)


def extract_smile(strikes, expiries, surface):
    strike_mesh, expiry_mesh, vol_surface = surface_mesh(strikes, expiries, surface)
    row = 0
    smile_strikes = strike_mesh[row, :] if strike_mesh.ndim == 2 else strike_mesh
    smile_vols = vol_surface[row, :] if vol_surface.ndim == 2 else vol_surface
    smile_days = expiry_mesh[row, 0] if expiry_mesh.ndim == 2 else expiry_mesh[row]
    return smile_strikes, smile_vols, float(smile_days)


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
        "points": int(np.size(vols)),
        "atm_term": atm_term,
        "strike_min": float(np.nanmin(strike_1d)),
        "strike_max": float(np.nanmax(strike_1d)),
    }
