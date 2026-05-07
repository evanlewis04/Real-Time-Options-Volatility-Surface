"""Optional SABR smile calibration for index/rates-style surfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


INDEX_STYLE_SYMBOLS = {"SPY", "QQQ", "IWM", "SPX", "NDX", "RUT"}


def hagan_sabr_iv(
    forward: np.ndarray | float,
    strike: np.ndarray | float,
    time_to_expiry: np.ndarray | float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Hagan lognormal SABR implied-volatility approximation."""
    f = np.asarray(forward, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(time_to_expiry, dtype=float)
    f = np.maximum(f, 1e-8)
    k = np.maximum(k, 1e-8)
    t = np.maximum(t, 1e-8)
    one_minus_beta = 1.0 - beta
    fk_beta = (f * k) ** (0.5 * one_minus_beta)
    log_fk = np.log(f / k)
    z = (nu / max(alpha, 1e-8)) * fk_beta * log_fk
    x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))
    z_over_x = np.ones_like(z, dtype=float)
    non_atm = np.abs(z) >= 1e-7
    z_over_x = np.where(non_atm, np.divide(z, x_z, out=z_over_x.copy(), where=non_atm), 1.0 - 0.5 * rho * z)
    denominator = fk_beta * (
        1.0
        + (one_minus_beta**2 / 24.0) * log_fk**2
        + (one_minus_beta**4 / 1920.0) * log_fk**4
    )
    correction = 1.0 + t * (
        (one_minus_beta**2 / 24.0) * alpha**2 / np.maximum((f * k) ** one_minus_beta, 1e-8)
        + 0.25 * rho * beta * nu * alpha / np.maximum(fk_beta, 1e-8)
        + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
    )
    return np.maximum((alpha / denominator) * z_over_x * correction, 1e-8)


def calibrate_sabr_by_expiry(
    chain: pd.DataFrame,
    spot: float,
    *,
    symbol: str | None = None,
    iv_column: str = "computedIV",
    beta: float = 1.0,
    min_points: int = 5,
) -> dict[str, Any]:
    """Fit optional SABR smiles by expiry without forcing them into equity UI."""
    if symbol and symbol.upper() not in INDEX_STYLE_SYMBOLS:
        return {
            "model": "SABR",
            "status": "skipped",
            "reason": "SABR is optional and currently shown only for index-style symbols.",
            "symbol": symbol.upper(),
            "smiles": [],
        }

    work = _prepared_frame(chain, spot, iv_column)
    if work.empty:
        return _empty_result("No valid IV rows for SABR calibration", symbol)

    smiles: list[dict[str, Any]] = []
    for expiry, group in work.groupby("expiration", dropna=True):
        smile = group.sort_values("strike")
        if len(smile) < min_points:
            continue
        forward = smile["forward"].to_numpy(dtype=float)
        strike = smile["strike"].to_numpy(dtype=float)
        time = smile["time"].to_numpy(dtype=float)
        observed = smile["iv"].to_numpy(dtype=float)
        params = _fit_sabr(forward, strike, time, observed, beta)
        fitted = hagan_sabr_iv(forward, strike, time, params["alpha"], beta, params["rho"], params["nu"])
        residuals = fitted - observed
        smiles.append(
            {
                "expiration": str(expiry),
                "dte": float(smile["dte"].median()),
                "points": int(len(smile)),
                "beta": float(beta),
                **params,
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "mae": float(np.mean(np.abs(residuals))),
                "max_error": float(np.max(np.abs(residuals))),
            }
        )

    if not smiles:
        return _empty_result("No expiry had enough valid points for SABR calibration", symbol)
    rmse = [row["rmse"] for row in smiles]
    return {
        "model": "SABR",
        "status": "fitted",
        "symbol": symbol.upper() if symbol else None,
        "beta": float(beta),
        "fitted_expiries": int(len(smiles)),
        "points": int(sum(row["points"] for row in smiles)),
        "rmse": float(np.mean(rmse)),
        "mae": float(np.mean([row["mae"] for row in smiles])),
        "max_error": float(max(row["max_error"] for row in smiles)),
        "smiles": smiles,
        "warnings": ["SABR is optional research analytics for index/rates-style smiles."],
    }


def _fit_sabr(
    forward: np.ndarray,
    strike: np.ndarray,
    time: np.ndarray,
    observed: np.ndarray,
    beta: float,
) -> dict[str, float]:
    atm = float(np.nanmedian(observed))

    def residual_vector(params: np.ndarray) -> np.ndarray:
        alpha, rho, nu = params
        return hagan_sabr_iv(forward, strike, time, alpha, beta, rho, nu) - observed

    result = least_squares(
        residual_vector,
        x0=np.array([max(atm, 0.01), -0.25, 0.50]),
        bounds=(np.array([1e-4, -0.95, 1e-4]), np.array([5.0, 0.95, 5.0])),
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=2000,
    )
    alpha, rho, nu = result.x
    return {"alpha": float(alpha), "rho": float(rho), "nu": float(nu)}


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
    if "forwardPrice" in chain:
        out["forward"] = pd.to_numeric(chain["forwardPrice"], errors="coerce")
    else:
        out["forward"] = float(spot)
    out["forward"] = out["forward"].where(out["forward"] > 0.0, float(spot))
    out["time"] = out["dte"] / 365.0
    out = out.dropna(subset=["expiration", "strike", "dte", "iv", "forward", "time"])
    return out[(out["strike"] > 0.0) & (out["time"] > 0.0) & (out["iv"] > 0.0)].copy()


def _empty_result(reason: str, symbol: str | None) -> dict[str, Any]:
    return {
        "model": "SABR",
        "status": "insufficient_data",
        "reason": reason,
        "symbol": symbol.upper() if symbol else None,
        "fitted_expiries": 0,
        "points": 0,
        "rmse": None,
        "mae": None,
        "max_error": None,
        "smiles": [],
    }
