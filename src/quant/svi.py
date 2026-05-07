"""SVI smile calibration utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Raw SVI total variance parameterization."""
    shifted = k - m
    return a + b * (rho * shifted + np.sqrt(shifted**2 + sigma**2))


def calibrate_svi_by_expiry(
    chain: pd.DataFrame,
    spot: float,
    *,
    iv_column: str = "computedIV",
    min_points: int = 5,
) -> pd.DataFrame:
    """Fit raw SVI parameters independently for each expiry."""
    if chain.empty or spot <= 0 or "expiration" not in chain.columns:
        return pd.DataFrame()
    work = chain.copy()
    if iv_column not in work:
        iv_column = "impliedVolatility"
    required = {"strike", "daysToExpiration", iv_column}
    if not required.issubset(work.columns):
        return pd.DataFrame()

    work["expiration_norm"] = pd.to_datetime(work["expiration"], errors="coerce").dt.normalize()
    work["iv_num"] = pd.to_numeric(work[iv_column], errors="coerce")
    work["strike_num"] = pd.to_numeric(work["strike"], errors="coerce")
    work["dte_num"] = pd.to_numeric(work["daysToExpiration"], errors="coerce")
    work["log_money_num"] = _log_moneyness(work, spot)
    work = work.dropna(subset=["expiration_norm", "iv_num", "strike_num", "dte_num", "log_money_num"])
    work = work[(work["iv_num"] > 0.0) & (work["dte_num"] > 0.0)]

    rows: list[dict[str, Any]] = []
    for expiry, group in work.groupby("expiration_norm", dropna=True):
        smile = group.sort_values("log_money_num")
        if len(smile) < min_points:
            continue
        dte = float(smile["dte_num"].median())
        t = max(dte / 365.0, 1e-8)
        k = smile["log_money_num"].to_numpy(dtype=float)
        observed_iv = smile["iv_num"].to_numpy(dtype=float)
        observed_w = observed_iv**2 * t
        params = _fit_svi(k, observed_w)
        fitted_w = np.maximum(svi_total_variance(k, **params), 1e-10)
        fitted_iv = np.sqrt(fitted_w / t)
        residuals = fitted_iv - observed_iv
        rows.append(
            {
                "expiration": expiry.date().isoformat(),
                "dte": dte,
                "points": int(len(smile)),
                **params,
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "mae": float(np.mean(np.abs(residuals))),
                "max_error": float(np.max(np.abs(residuals))),
                "residuals": [
                    {
                        "log_moneyness": float(log_money),
                        "strike": float(strike),
                        "observed_iv": float(observed),
                        "fitted_iv": float(fitted),
                        "residual": float(residual),
                    }
                    for log_money, strike, observed, fitted, residual in zip(
                        k,
                        smile["strike_num"].to_numpy(dtype=float),
                        observed_iv,
                        fitted_iv,
                        residuals,
                    )
                ],
            }
        )
    return pd.DataFrame(rows).sort_values("dte").reset_index(drop=True) if rows else pd.DataFrame()


def fit_diagnostics_from_svi(svi_rows: pd.DataFrame) -> dict[str, Any]:
    """Summarize per-expiry SVI fit quality for surface metadata."""
    if svi_rows.empty:
        return {
            "model": "SVI",
            "fitted_expiries": 0,
            "rmse": None,
            "mae": None,
            "max_error": None,
            "points": 0,
        }
    return {
        "model": "SVI",
        "fitted_expiries": int(len(svi_rows)),
        "rmse": float(pd.to_numeric(svi_rows["rmse"], errors="coerce").mean()),
        "mae": float(pd.to_numeric(svi_rows["mae"], errors="coerce").mean()),
        "max_error": float(pd.to_numeric(svi_rows["max_error"], errors="coerce").max()),
        "points": int(pd.to_numeric(svi_rows["points"], errors="coerce").sum()),
    }


def _fit_svi(k: np.ndarray, observed_w: np.ndarray) -> dict[str, float]:
    min_w = max(float(np.nanmin(observed_w)), 1e-6)
    max_w = max(float(np.nanmax(observed_w)), min_w)
    x0 = np.array([min_w * 0.5, max(max_w, 1e-4), 0.0, float(np.median(k)), 0.1])
    lower = np.array([0.0, 1e-8, -0.999, -2.0, 1e-4])
    upper = np.array([5.0, 10.0, 0.999, 2.0, 5.0])
    result = least_squares(
        lambda params: svi_total_variance(k, *params) - observed_w,
        x0=np.clip(x0, lower, upper),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=2000,
    )
    a, b, rho, m, sigma = result.x
    return {
        "a": float(a),
        "b": float(b),
        "rho": float(rho),
        "m": float(m),
        "sigma": float(sigma),
    }


def _log_moneyness(work: pd.DataFrame, spot: float) -> pd.Series:
    if "logMoneyness" in work:
        out = pd.to_numeric(work["logMoneyness"], errors="coerce")
        if out.notna().any():
            return out
    if "forwardPrice" in work:
        forwards = pd.to_numeric(work["forwardPrice"], errors="coerce")
        strikes = pd.to_numeric(work["strike"], errors="coerce")
        return np.log(strikes / forwards.where(forwards > 0))
    strikes = pd.to_numeric(work["strike"], errors="coerce")
    return np.log(strikes / float(spot))
