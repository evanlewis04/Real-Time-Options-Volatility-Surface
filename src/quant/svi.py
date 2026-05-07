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


def ssvi_total_variance(k: np.ndarray, theta: np.ndarray, rho: float, eta: float, gamma: float) -> np.ndarray:
    """Surface SVI total variance with power-law phi(theta)."""
    theta_safe = np.maximum(np.asarray(theta, dtype=float), 1e-10)
    phi = eta / np.power(theta_safe, gamma)
    scaled_log_money = phi * np.asarray(k, dtype=float)
    return 0.5 * theta_safe * (
        1.0
        + rho * scaled_log_money
        + np.sqrt((scaled_log_money + rho) ** 2 + 1.0 - rho**2)
    )


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


def calibrate_ssvi_surface(
    chain: pd.DataFrame,
    spot: float,
    *,
    iv_column: str = "computedIV",
    min_expiries: int = 2,
    min_points_per_expiry: int = 5,
) -> dict[str, Any]:
    """Fit a constrained global SSVI surface across expiries.

    The term structure theta is estimated from the near-ATM total variance for
    each expiry, then forced to be nondecreasing before fitting a single
    power-law SSVI smile shape across all expiries.
    """
    work = _prepared_surface_frame(chain, spot, iv_column)
    if work.empty:
        return _empty_ssvi_result("No valid IV rows for SSVI calibration")

    expiry_rows = _expiry_theta_rows(work, min_points_per_expiry)
    if len(expiry_rows) < min_expiries:
        return _empty_ssvi_result("Fewer than two expiries have enough valid IV points")

    expiry_frame = pd.DataFrame(expiry_rows).sort_values("dte").reset_index(drop=True)
    expiry_frame["theta"] = np.maximum.accumulate(expiry_frame["raw_theta"].to_numpy(dtype=float))
    theta_by_expiry = dict(zip(expiry_frame["expiration"], expiry_frame["theta"]))
    fit_rows = work[work["expiration_norm"].dt.date.astype(str).isin(theta_by_expiry)].copy()
    fit_rows["theta"] = fit_rows["expiration_norm"].dt.date.astype(str).map(theta_by_expiry)
    fit_rows = fit_rows.dropna(subset=["theta"])
    if fit_rows.empty:
        return _empty_ssvi_result("No rows matched calibrated SSVI expiries")

    params = _fit_ssvi(
        fit_rows["log_money_num"].to_numpy(dtype=float),
        fit_rows["iv_num"].to_numpy(dtype=float),
        fit_rows["time_num"].to_numpy(dtype=float),
        fit_rows["theta"].to_numpy(dtype=float),
    )
    fitted_w = np.maximum(
        ssvi_total_variance(
            fit_rows["log_money_num"].to_numpy(dtype=float),
            fit_rows["theta"].to_numpy(dtype=float),
            params["rho"],
            params["eta"],
            params["gamma"],
        ),
        1e-10,
    )
    fitted_iv = np.sqrt(fitted_w / fit_rows["time_num"].to_numpy(dtype=float))
    observed_iv = fit_rows["iv_num"].to_numpy(dtype=float)
    residuals = fitted_iv - observed_iv
    constraints = ssvi_constraint_summary(
        expiry_frame["theta"].to_numpy(dtype=float),
        params["rho"],
        params["eta"],
        params["gamma"],
    )
    return {
        "model": "SSVI",
        "status": "fitted",
        "parameterization": "surface_svi_power_law_phi",
        "documented_constraints": [
            "theta is nondecreasing by expiry",
            "theta * phi(theta) is nondecreasing by expiry",
            "theta * phi(theta) * (1 + |rho|) <= 4",
            "theta * phi(theta)^2 * (1 + |rho|) <= 4",
        ],
        "rho": params["rho"],
        "eta": params["eta"],
        "gamma": params["gamma"],
        "fitted_expiries": int(len(expiry_frame)),
        "points": int(len(fit_rows)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "max_error": float(np.max(np.abs(residuals))),
        "constraints": constraints,
        "atm_total_variance": [
            {
                "expiration": str(row.expiration),
                "dte": float(row.dte),
                "theta": float(row.theta),
                "raw_theta": float(row.raw_theta),
                "points": int(row.points),
            }
            for row in expiry_frame.itertuples()
        ],
        "residuals": [
            {
                "expiration": expiry.date().isoformat(),
                "dte": float(dte),
                "log_moneyness": float(log_money),
                "strike": float(strike),
                "observed_iv": float(observed),
                "fitted_iv": float(fitted),
                "residual": float(residual),
            }
            for expiry, dte, log_money, strike, observed, fitted, residual in zip(
                fit_rows["expiration_norm"],
                fit_rows["dte_num"],
                fit_rows["log_money_num"],
                fit_rows["strike_num"],
                observed_iv,
                fitted_iv,
                residuals,
            )
        ],
    }


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


def fit_diagnostics_from_ssvi(ssvi_result: dict[str, Any]) -> dict[str, Any]:
    """Summarize global SSVI fit quality for surface metadata."""
    return {
        "model": "SSVI",
        "status": ssvi_result.get("status", "unavailable"),
        "fitted_expiries": int(ssvi_result.get("fitted_expiries") or 0),
        "points": int(ssvi_result.get("points") or 0),
        "rmse": ssvi_result.get("rmse"),
        "mae": ssvi_result.get("mae"),
        "max_error": ssvi_result.get("max_error"),
        "constraints_passed": bool((ssvi_result.get("constraints") or {}).get("passed", False)),
    }


def ssvi_constraint_summary(theta: np.ndarray, rho: float, eta: float, gamma: float) -> dict[str, Any]:
    """Return no-arbitrage-oriented diagnostics for the SSVI parameter set."""
    theta_safe = np.maximum(np.asarray(theta, dtype=float), 1e-10)
    phi = eta / np.power(theta_safe, gamma)
    theta_phi = theta_safe * phi
    one_plus_abs_rho = 1.0 + abs(float(rho))
    butterfly_slope = theta_phi * one_plus_abs_rho
    butterfly_curvature = theta_safe * phi**2 * one_plus_abs_rho
    tolerance = 1e-8
    calendar_theta = bool(np.all(np.diff(theta_safe) >= -tolerance))
    calendar_theta_phi = bool(np.all(np.diff(theta_phi) >= -tolerance))
    butterfly_slope_ok = bool(np.nanmax(butterfly_slope) <= 4.0 + tolerance)
    butterfly_curvature_ok = bool(np.nanmax(butterfly_curvature) <= 4.0 + tolerance)
    return {
        "passed": bool(calendar_theta and calendar_theta_phi and butterfly_slope_ok and butterfly_curvature_ok),
        "calendar_theta_monotonic": calendar_theta,
        "calendar_theta_phi_monotonic": calendar_theta_phi,
        "butterfly_slope_bound": butterfly_slope_ok,
        "butterfly_curvature_bound": butterfly_curvature_ok,
        "max_theta_phi_one_plus_abs_rho": float(np.nanmax(butterfly_slope)),
        "max_theta_phi_squared_one_plus_abs_rho": float(np.nanmax(butterfly_curvature)),
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


def _fit_ssvi(k: np.ndarray, observed_iv: np.ndarray, time: np.ndarray, theta: np.ndarray) -> dict[str, float]:
    def residual_vector(params: np.ndarray) -> np.ndarray:
        rho, eta, gamma = params
        fitted_w = np.maximum(ssvi_total_variance(k, theta, rho, eta, gamma), 1e-10)
        fitted_iv = np.sqrt(fitted_w / time)
        residuals = fitted_iv - observed_iv
        constraints = ssvi_constraint_summary(np.unique(theta), rho, eta, gamma)
        penalties = np.array(
            [
                max(0.0, constraints["max_theta_phi_one_plus_abs_rho"] - 4.0),
                max(0.0, constraints["max_theta_phi_squared_one_plus_abs_rho"] - 4.0),
            ],
            dtype=float,
        )
        return np.concatenate([residuals, penalties * 10.0])

    result = least_squares(
        residual_vector,
        x0=np.array([-0.25, 1.0, 0.25]),
        bounds=(np.array([-0.95, 1e-4, 0.0]), np.array([0.95, 10.0, 0.5])),
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=3000,
    )
    rho, eta, gamma = result.x
    return {"rho": float(rho), "eta": float(eta), "gamma": float(gamma)}


def _prepared_surface_frame(chain: pd.DataFrame, spot: float, iv_column: str) -> pd.DataFrame:
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
    work["time_num"] = work["dte_num"] / 365.0
    work["log_money_num"] = _log_moneyness(work, spot)
    work = work.dropna(
        subset=["expiration_norm", "iv_num", "strike_num", "dte_num", "time_num", "log_money_num"]
    )
    return work[(work["iv_num"] > 0.0) & (work["time_num"] > 0.0)].copy()


def _expiry_theta_rows(work: pd.DataFrame, min_points: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expiry, group in work.groupby("expiration_norm", dropna=True):
        if len(group) < min_points:
            continue
        group = group.copy()
        group["atm_distance"] = group["log_money_num"].abs()
        sample = group.sort_values("atm_distance").head(min(3, len(group)))
        dte = float(group["dte_num"].median())
        time = max(dte / 365.0, 1e-8)
        raw_theta = float(np.median(sample["iv_num"].to_numpy(dtype=float) ** 2 * time))
        if not np.isfinite(raw_theta) or raw_theta <= 0:
            continue
        rows.append(
            {
                "expiration": expiry.date().isoformat(),
                "dte": dte,
                "raw_theta": raw_theta,
                "points": int(len(group)),
            }
        )
    return rows


def _empty_ssvi_result(reason: str) -> dict[str, Any]:
    return {
        "model": "SSVI",
        "status": "insufficient_data",
        "reason": reason,
        "parameterization": "surface_svi_power_law_phi",
        "fitted_expiries": 0,
        "points": 0,
        "rmse": None,
        "mae": None,
        "max_error": None,
        "constraints": {"passed": False},
        "atm_total_variance": [],
        "residuals": [],
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
