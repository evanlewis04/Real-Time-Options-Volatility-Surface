"""Research-only Heston calibration diagnostics.

This module deliberately does not expose production Heston option pricing. It
fits a compact variance-dynamics surrogate to stored or current IV snapshots so
the dashboard can report fit errors and warnings while keeping model provenance
honest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.data.models import MarketDataSnapshot
from src.data.snapshots import load_snapshot


def heston_research_total_variance(
    log_moneyness: np.ndarray,
    time_to_expiry: np.ndarray,
    v0: float,
    theta: float,
    kappa: float,
    rho: float,
    vol_of_var: float,
) -> np.ndarray:
    """Approximate Heston-style total variance for calibration diagnostics."""
    t = np.maximum(np.asarray(time_to_expiry, dtype=float), 1e-8)
    k = np.asarray(log_moneyness, dtype=float)
    mean_reversion = (1.0 - np.exp(-kappa * t)) / np.maximum(kappa * t, 1e-8)
    average_variance = theta + (v0 - theta) * mean_reversion
    skew = np.maximum(0.10, 1.0 + rho * vol_of_var * k / (1.0 + np.abs(k)))
    curvature = 1.0 + 0.25 * vol_of_var**2 * k**2
    return np.maximum(t * average_variance * skew * curvature, 1e-10)


def calibrate_heston_research(
    chain: pd.DataFrame,
    spot: float,
    *,
    iv_column: str = "computedIV",
    min_points: int = 8,
) -> dict[str, Any]:
    """Fit research Heston diagnostics to a normalized option-chain frame."""
    work = _prepared_frame(chain, spot, iv_column)
    if len(work) < min_points:
        return _empty_result("Fewer than eight valid IV points for Heston research calibration")

    observed_w = work["iv"].to_numpy(dtype=float) ** 2 * work["time"].to_numpy(dtype=float)
    initial_var = max(float(np.nanmedian(work["iv"] ** 2)), 1e-4)
    result = least_squares(
        lambda params: (
            heston_research_total_variance(
                work["log_moneyness"].to_numpy(dtype=float),
                work["time"].to_numpy(dtype=float),
                params[0],
                params[1],
                params[2],
                params[3],
                params[4],
            )
            - observed_w
        ),
        x0=np.array([initial_var, initial_var, 1.0, -0.35, 0.50]),
        bounds=(
            np.array([1e-6, 1e-6, 0.05, -0.95, 0.01]),
            np.array([4.0, 4.0, 10.0, 0.95, 5.0]),
        ),
        loss="soft_l1",
        f_scale=0.001,
        max_nfev=2000,
    )
    fitted_w = heston_research_total_variance(
        work["log_moneyness"].to_numpy(dtype=float),
        work["time"].to_numpy(dtype=float),
        result.x[0],
        result.x[1],
        result.x[2],
        result.x[3],
        result.x[4],
    )
    fitted_iv = np.sqrt(fitted_w / work["time"].to_numpy(dtype=float))
    residuals = fitted_iv - work["iv"].to_numpy(dtype=float)
    v0, theta, kappa, rho, vol_of_var = result.x
    return {
        "model": "Heston research",
        "status": "fitted",
        "parameterization": "variance_dynamics_surrogate",
        "warnings": [
            "Research calibration only; not a production Heston characteristic-function pricer.",
            "Use fit errors as diagnostics, not tradable model values.",
        ],
        "points": int(len(work)),
        "fitted_expiries": int(work["expiration"].nunique()),
        "v0": float(v0),
        "theta": float(theta),
        "kappa": float(kappa),
        "rho": float(rho),
        "vol_of_var": float(vol_of_var),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "max_error": float(np.max(np.abs(residuals))),
        "residuals": [
            {
                "expiration": str(expiry),
                "dte": float(dte),
                "log_moneyness": float(log_money),
                "observed_iv": float(observed),
                "fitted_iv": float(fitted),
                "residual": float(residual),
            }
            for expiry, dte, log_money, observed, fitted, residual in zip(
                work["expiration"],
                work["dte"],
                work["log_moneyness"],
                work["iv"],
                fitted_iv,
                residuals,
            )
        ],
    }


def calibrate_heston_from_snapshot(
    snapshot: MarketDataSnapshot | str | Path,
    *,
    iv_column: str = "computedIV",
) -> dict[str, Any]:
    """Run Heston research calibration on a persisted snapshot or snapshot object."""
    loaded = load_snapshot(snapshot) if isinstance(snapshot, (str, Path)) else snapshot
    result = calibrate_heston_research(loaded.options_frame(), loaded.spot, iv_column=iv_column)
    return {
        **result,
        "snapshot_symbol": loaded.symbol,
        "snapshot_timestamp": loaded.spot_timestamp.isoformat(),
        "snapshot_source": loaded.source,
        "snapshot_mode": loaded.mode,
    }


def _prepared_frame(chain: pd.DataFrame, spot: float, iv_column: str) -> pd.DataFrame:
    if chain.empty or spot <= 0:
        return pd.DataFrame()
    if iv_column not in chain:
        iv_column = "impliedVolatility"
    required = {"expiration", "strike", "daysToExpiration", iv_column}
    if not required.issubset(chain.columns):
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "expiration": pd.to_datetime(chain.get("expiration"), errors="coerce").dt.date.astype(str),
            "strike": pd.to_numeric(chain.get("strike"), errors="coerce"),
            "dte": pd.to_numeric(chain.get("daysToExpiration"), errors="coerce"),
            "iv": pd.to_numeric(chain.get(iv_column), errors="coerce"),
        }
    )
    if "logMoneyness" in chain:
        out["log_moneyness"] = pd.to_numeric(chain["logMoneyness"], errors="coerce")
    else:
        out["log_moneyness"] = np.nan
    fallback_log_money = np.log(out["strike"] / float(spot))
    out["log_moneyness"] = out["log_moneyness"].where(out["log_moneyness"].notna(), fallback_log_money)
    out["time"] = out["dte"] / 365.0
    out = out.dropna(subset=["strike", "dte", "iv", "log_moneyness", "time"])
    return out[(out["strike"] > 0.0) & (out["time"] > 0.0) & (out["iv"] > 0.0)].copy()


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "model": "Heston research",
        "status": "insufficient_data",
        "reason": reason,
        "parameterization": "variance_dynamics_surrogate",
        "points": 0,
        "fitted_expiries": 0,
        "rmse": None,
        "mae": None,
        "max_error": None,
        "warnings": ["Research calibration only; Heston production pricing is not enabled."],
        "residuals": [],
    }
