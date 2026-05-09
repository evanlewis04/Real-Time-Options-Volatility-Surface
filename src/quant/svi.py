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
    weight_column: str | None = "fitWeight",
    use_weight_fallbacks: bool = True,
    loss: str = "soft_l1",
    loss_f_scale: float = 0.01,
    min_points: int = 5,
) -> pd.DataFrame:
    """Fit raw SVI parameters independently for each expiry."""
    if chain.empty or spot <= 0 or "expiration" not in chain.columns:
        return pd.DataFrame()
    loss_mode = _validate_loss(loss)
    f_scale = _validate_loss_f_scale(loss_f_scale)
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
        weights, weight_meta = _svi_row_weights(
            smile,
            weight_column=weight_column,
            use_fallbacks=use_weight_fallbacks,
        )
        params = _fit_svi(k, observed_w, sample_weights=weights, loss=loss_mode, loss_f_scale=f_scale)
        fitted_w = np.maximum(svi_total_variance(k, **params), 1e-10)
        fitted_iv = np.sqrt(fitted_w / t)
        residuals = fitted_iv - observed_iv
        weighted_rmse = _weighted_rmse(residuals, weights)
        residual_rows = [
            {
                "log_moneyness": float(log_money),
                "strike": float(strike),
                "observed_iv": float(observed),
                "fitted_iv": float(fitted),
                "residual": float(residual),
                "fit_weight": float(weight),
            }
            for log_money, strike, observed, fitted, residual, weight in zip(
                k,
                smile["strike_num"].to_numpy(dtype=float),
                observed_iv,
                fitted_iv,
                residuals,
                weights,
            )
        ]
        residual_diagnostics = _residual_diagnostics(
            residual_rows,
            model="SVI",
            loss_mode=loss_mode,
            loss_f_scale=f_scale,
        )
        rows.append(
            {
                "expiration": expiry.date().isoformat(),
                "dte": dte,
                "points": int(len(smile)),
                **params,
                **weight_meta,
                "loss_mode": loss_mode,
                "loss_f_scale": f_scale,
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "weighted_rmse": weighted_rmse,
                "mae": float(np.mean(np.abs(residuals))),
                "max_error": float(np.max(np.abs(residuals))),
                "residual_diagnostics": residual_diagnostics,
                "residuals": residual_rows,
            }
        )
    return pd.DataFrame(rows).sort_values("dte").reset_index(drop=True) if rows else pd.DataFrame()


def calibrate_ssvi_surface(
    chain: pd.DataFrame,
    spot: float,
    *,
    iv_column: str = "computedIV",
    weight_column: str | None = "fitWeight",
    use_weight_fallbacks: bool = True,
    loss: str = "soft_l1",
    loss_f_scale: float = 0.01,
    min_expiries: int = 2,
    min_points_per_expiry: int = 5,
) -> dict[str, Any]:
    """Fit a constrained global SSVI surface across expiries.

    The term structure theta is estimated from the near-ATM total variance for
    each expiry, then forced to be nondecreasing before fitting a single
    power-law SSVI smile shape across all expiries.
    """
    loss_mode = _validate_loss(loss)
    f_scale = _validate_loss_f_scale(loss_f_scale)
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

    weights, weight_meta = _svi_row_weights(
        fit_rows,
        weight_column=weight_column,
        use_fallbacks=use_weight_fallbacks,
    )
    params = _fit_ssvi(
        fit_rows["log_money_num"].to_numpy(dtype=float),
        fit_rows["iv_num"].to_numpy(dtype=float),
        fit_rows["time_num"].to_numpy(dtype=float),
        fit_rows["theta"].to_numpy(dtype=float),
        sample_weights=weights,
        loss=loss_mode,
        loss_f_scale=f_scale,
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
    unweighted_rmse = float(np.sqrt(np.mean(residuals**2)))
    weighted_rmse = _weighted_rmse(residuals, weights)
    constraints = ssvi_constraint_summary(
        expiry_frame["theta"].to_numpy(dtype=float),
        params["rho"],
        params["eta"],
        params["gamma"],
    )
    residual_rows = [
        {
            "expiration": expiry.date().isoformat(),
            "dte": float(dte),
            "log_moneyness": float(log_money),
            "strike": float(strike),
            "observed_iv": float(observed),
            "fitted_iv": float(fitted),
            "residual": float(residual),
            "fit_weight": float(weight),
        }
        for expiry, dte, log_money, strike, observed, fitted, residual, weight in zip(
            fit_rows["expiration_norm"],
            fit_rows["dte_num"],
            fit_rows["log_money_num"],
            fit_rows["strike_num"],
            observed_iv,
            fitted_iv,
            residuals,
            weights,
        )
    ]
    return {
        "model": "SSVI",
        "status": "fitted",
        "parameterization": "surface_svi_power_law_phi",
        "loss_mode": loss_mode,
        "loss_f_scale": f_scale,
        **weight_meta,
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
        "rmse": unweighted_rmse,
        "unweighted_rmse": unweighted_rmse,
        "weighted_rmse": weighted_rmse,
        "mae": float(np.mean(np.abs(residuals))),
        "max_error": float(np.max(np.abs(residuals))),
        "constraints": constraints,
        "residual_diagnostics": _residual_diagnostics(
            residual_rows,
            model="SSVI",
            loss_mode=loss_mode,
            loss_f_scale=f_scale,
        ),
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
        "residuals": residual_rows,
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
            "residual_diagnostics": _empty_residual_diagnostics("SVI"),
        }
    residual_diagnostics = _residual_diagnostics(
        _flatten_svi_residual_rows(svi_rows),
        model="SVI",
        loss_mode=_joined_unique(svi_rows, "loss_mode"),
        loss_f_scale=_mean_or_none(svi_rows, "loss_f_scale"),
    )
    return {
        "model": "SVI",
        "fitted_expiries": int(len(svi_rows)),
        "rmse": float(pd.to_numeric(svi_rows["rmse"], errors="coerce").mean()),
        "weighted_rmse": _mean_or_none(svi_rows, "weighted_rmse"),
        "mae": float(pd.to_numeric(svi_rows["mae"], errors="coerce").mean()),
        "max_error": float(pd.to_numeric(svi_rows["max_error"], errors="coerce").max()),
        "points": int(pd.to_numeric(svi_rows["points"], errors="coerce").sum()),
        "weight_mode": _joined_unique(svi_rows, "weight_mode"),
        "weight_column": _joined_unique(svi_rows, "weight_column"),
        "loss_mode": _joined_unique(svi_rows, "loss_mode"),
        "loss_f_scale": _mean_or_none(svi_rows, "loss_f_scale"),
        "residual_diagnostics": residual_diagnostics,
    }


def fit_diagnostics_from_ssvi(ssvi_result: dict[str, Any]) -> dict[str, Any]:
    """Summarize global SSVI fit quality for surface metadata."""
    return {
        "model": "SSVI",
        "status": ssvi_result.get("status", "unavailable"),
        "fitted_expiries": int(ssvi_result.get("fitted_expiries") or 0),
        "points": int(ssvi_result.get("points") or 0),
        "rmse": ssvi_result.get("rmse"),
        "unweighted_rmse": ssvi_result.get("unweighted_rmse"),
        "weighted_rmse": ssvi_result.get("weighted_rmse"),
        "mae": ssvi_result.get("mae"),
        "max_error": ssvi_result.get("max_error"),
        "weight_mode": ssvi_result.get("weight_mode"),
        "weight_column": ssvi_result.get("weight_column"),
        "loss_mode": ssvi_result.get("loss_mode"),
        "loss_f_scale": ssvi_result.get("loss_f_scale"),
        "residual_diagnostics": ssvi_result.get("residual_diagnostics") or _empty_residual_diagnostics("SSVI"),
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


def _fit_svi(
    k: np.ndarray,
    observed_w: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    loss: str,
    loss_f_scale: float,
) -> dict[str, float]:
    min_w = max(float(np.nanmin(observed_w)), 1e-6)
    max_w = max(float(np.nanmax(observed_w)), min_w)
    x0 = np.array([min_w * 0.5, max(max_w, 1e-4), 0.0, float(np.median(k)), 0.1])
    lower = np.array([0.0, 1e-8, -0.999, -2.0, 1e-4])
    upper = np.array([5.0, 10.0, 0.999, 2.0, 5.0])
    sqrt_weights = _least_squares_sqrt_weights(sample_weights, len(k))
    result = least_squares(
        lambda params: (svi_total_variance(k, *params) - observed_w) * sqrt_weights,
        x0=np.clip(x0, lower, upper),
        bounds=(lower, upper),
        loss=loss,
        f_scale=loss_f_scale,
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


def _fit_ssvi(
    k: np.ndarray,
    observed_iv: np.ndarray,
    time: np.ndarray,
    theta: np.ndarray,
    *,
    sample_weights: np.ndarray | None = None,
    loss: str,
    loss_f_scale: float,
) -> dict[str, float]:
    sqrt_weights = _least_squares_sqrt_weights(sample_weights, len(k))

    def residual_vector(params: np.ndarray) -> np.ndarray:
        rho, eta, gamma = params
        fitted_w = np.maximum(ssvi_total_variance(k, theta, rho, eta, gamma), 1e-10)
        fitted_iv = np.sqrt(fitted_w / time)
        residuals = (fitted_iv - observed_iv) * sqrt_weights
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
        loss=loss,
        f_scale=loss_f_scale,
        max_nfev=3000,
    )
    rho, eta, gamma = result.x
    return {"rho": float(rho), "eta": float(eta), "gamma": float(gamma)}


def _svi_row_weights(
    smile: pd.DataFrame,
    *,
    weight_column: str | None,
    use_fallbacks: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return deterministic per-row SVI weights plus provenance metadata."""
    source_column = _first_numeric_column(smile, [weight_column] if weight_column else [])
    mode = "uniform"
    if source_column:
        raw = pd.to_numeric(smile[source_column], errors="coerce").to_numpy(dtype=float)
        mode = "quote_reliability_liquidity" if source_column == "fitWeight" else "provided"
    elif use_fallbacks:
        reliability_column = _first_numeric_column(smile, ["quoteReliabilityScore"])
        liquidity_weights = _liquidity_weights(smile)
        if reliability_column:
            reliability = pd.to_numeric(smile[reliability_column], errors="coerce").to_numpy(dtype=float)
            if liquidity_weights is None:
                liquidity_weights = np.ones(len(smile), dtype=float)
            raw = reliability * liquidity_weights
            source_column = reliability_column
            mode = "quote_reliability_liquidity"
        elif liquidity_weights is not None:
            raw = liquidity_weights
            mode = "liquidity"
        else:
            raw = np.ones(len(smile), dtype=float)
    else:
        raw = np.ones(len(smile), dtype=float)

    weights = _sanitize_weights(raw, len(smile))
    return weights, _weight_metadata(weights, mode=mode, column=source_column)


def _first_numeric_column(frame: pd.DataFrame, columns: list[str | None]) -> str | None:
    for column in columns:
        if column and column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return column
    return None


def _liquidity_weights(smile: pd.DataFrame) -> np.ndarray | None:
    columns = [column for column in ("volume", "openInterest") if column in smile]
    if not columns:
        return None
    components: list[np.ndarray] = []
    for column in columns:
        values = pd.to_numeric(smile[column], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        scale = float(np.nanmedian(values[values > 0])) if np.any(values > 0) else 0.0
        if scale <= 0.0 or not np.isfinite(scale):
            components.append(np.full(len(smile), 0.5, dtype=float))
        else:
            components.append(np.clip(values / scale, 0.05, 2.0))
    return np.mean(components, axis=0)


def _sanitize_weights(raw: np.ndarray, size: int) -> np.ndarray:
    if raw.size != size:
        return np.ones(size, dtype=float)
    weights = np.asarray(raw, dtype=float)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    if not np.any(weights > 0.0):
        return np.ones(size, dtype=float)
    return weights


def _least_squares_sqrt_weights(sample_weights: np.ndarray | None, size: int) -> np.ndarray:
    weights = _sanitize_weights(sample_weights, size) if sample_weights is not None else np.ones(size, dtype=float)
    positive = weights[weights > 0.0]
    normalized = weights / float(np.mean(positive))
    return np.sqrt(np.clip(normalized, 0.0, None))


def _validate_loss(loss: str) -> str:
    normalized = str(loss).strip().lower()
    if normalized not in {"linear", "huber", "soft_l1"}:
        raise ValueError(f"Unsupported SVI loss mode: {loss!r}")
    return normalized


def _validate_loss_f_scale(loss_f_scale: float) -> float:
    value = float(loss_f_scale)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("SVI loss_f_scale must be a positive finite number")
    return value


def _weight_metadata(weights: np.ndarray, *, mode: str, column: str | None) -> dict[str, Any]:
    positive = weights[weights > 0.0]
    return {
        "weight_mode": mode,
        "weight_column": column,
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
        "positive_weight_count": int(len(positive)),
    }


def _weighted_rmse(residuals: np.ndarray, weights: np.ndarray) -> float:
    clean_weights = _sanitize_weights(weights, len(residuals))
    total_weight = float(np.sum(clean_weights))
    if total_weight <= 0.0:
        return float(np.sqrt(np.mean(residuals**2)))
    return float(np.sqrt(np.average(residuals**2, weights=clean_weights)))


def _flatten_svi_residual_rows(svi_rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for smile in svi_rows.to_dict("records"):
        for row in smile.get("residuals") or []:
            rows.append(
                {
                    **row,
                    "expiration": smile.get("expiration"),
                    "dte": smile.get("dte"),
                    "model": "SVI",
                }
            )
    return rows


def _residual_diagnostics(
    rows: list[dict[str, Any]],
    *,
    model: str,
    loss_mode: str | None,
    loss_f_scale: float | None,
    top_n: int = 10,
) -> dict[str, Any]:
    if not rows:
        return _empty_residual_diagnostics(model)

    frame = pd.DataFrame(rows)
    frame["residual"] = pd.to_numeric(frame.get("residual"), errors="coerce")
    frame["fit_weight"] = pd.to_numeric(frame.get("fit_weight", 1.0), errors="coerce").fillna(1.0)
    frame = frame.dropna(subset=["residual"])
    if frame.empty:
        return _empty_residual_diagnostics(model)

    residuals = frame["residual"].to_numpy(dtype=float)
    abs_residuals = np.abs(residuals)
    threshold = _residual_clip_threshold(residuals)
    clipped = np.clip(residuals, -threshold, threshold)
    clipped_mask = abs_residuals > threshold
    positive_weights = frame.loc[frame["fit_weight"] > 0.0, "fit_weight"]
    weight_threshold = float(positive_weights.median() * 0.5) if not positive_weights.empty else 0.0
    downweighted_mask = frame["fit_weight"].to_numpy(dtype=float) < weight_threshold if weight_threshold > 0 else np.zeros(
        len(frame),
        dtype=bool,
    )

    display = frame.copy()
    display["abs_residual"] = abs_residuals
    display["clipped_residual"] = clipped
    display["clipped"] = clipped_mask
    display["downweighted"] = downweighted_mask
    top_rows = display.sort_values("abs_residual", ascending=False).head(top_n)

    return {
        "model": model,
        "policy": "diagnostic_only_no_rows_removed",
        "loss_mode": loss_mode,
        "loss_f_scale": loss_f_scale,
        "points": int(len(frame)),
        "clip_threshold_abs_residual": float(threshold),
        "clipped_count": int(np.count_nonzero(clipped_mask)),
        "downweighted_count": int(np.count_nonzero(downweighted_mask)),
        "downweight_threshold": weight_threshold,
        "rmse_before_clipping": float(np.sqrt(np.mean(residuals**2))),
        "rmse_after_clipping": float(np.sqrt(np.mean(clipped**2))),
        "rmse_clipping_impact": float(np.sqrt(np.mean(residuals**2)) - np.sqrt(np.mean(clipped**2))),
        "max_abs_residual": float(np.max(abs_residuals)),
        "top_residuals": [_top_residual_row(row) for row in top_rows.to_dict("records")],
    }


def _residual_clip_threshold(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=float)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_sigma = 1.4826 * mad
    if np.isfinite(robust_sigma) and robust_sigma > 0.0:
        return max(3.0 * robust_sigma, 1e-6)
    abs_residuals = np.abs(residuals)
    fallback = float(np.quantile(abs_residuals, 0.95)) if len(abs_residuals) else 0.0
    return max(fallback, 1e-6)


def _top_residual_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model",
        "expiration",
        "dte",
        "strike",
        "log_moneyness",
        "observed_iv",
        "fitted_iv",
        "residual",
        "abs_residual",
        "clipped_residual",
        "fit_weight",
        "clipped",
        "downweighted",
    )
    return {key: _json_safe_value(row.get(key)) for key in keys if key in row}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _empty_residual_diagnostics(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "policy": "diagnostic_only_no_rows_removed",
        "points": 0,
        "clip_threshold_abs_residual": None,
        "clipped_count": 0,
        "downweighted_count": 0,
        "downweight_threshold": None,
        "rmse_before_clipping": None,
        "rmse_after_clipping": None,
        "rmse_clipping_impact": None,
        "max_abs_residual": None,
        "top_residuals": [],
    }


def _mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _joined_unique(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame:
        return None
    values = [str(value) for value in frame[column].dropna().unique() if str(value)]
    return ",".join(sorted(values)) if values else None


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
