"""Fit-mode validation and deterministic surface backtesting diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data.models import MarketDataSnapshot
from src.quant.surface_arbitrage import check_surface_arbitrage
from src.quant.svi import (
    calibrate_ssvi_surface,
    calibrate_svi_by_expiry,
    ssvi_total_variance,
    svi_total_variance,
)


VALIDATION_PROVENANCE = "fit_mode_validation_diagnostic_not_market_observation"
BACKTEST_PROVENANCE = "fit_mode_backtest_diagnostic_not_market_observation"
PRIOR_ASSISTED_PROVENANCE = "prior_assisted_fit_estimate_not_market_observation"


def validate_fit_modes(
    chain: pd.DataFrame,
    spot: float,
    *,
    iv_column: str = "computedIV",
    baseline_chain: pd.DataFrame | None = None,
    holdout_stride: int = 3,
) -> dict[str, Any]:
    """Return deterministic validation diagnostics for supported fit modes.

    The residuals compare fitted estimates with normalized fixture or provider
    IV inputs. They are diagnostics, not new market observations.
    """
    prepared = _prepared_validation_frame(chain, spot, iv_column)
    if prepared.empty:
        return _unavailable_validation("Current chain has no usable IV rows")

    train, holdout = _train_holdout_split(prepared, holdout_stride)
    if train.empty or holdout.empty:
        return _unavailable_validation("Need both training and holdout rows for validation")

    baseline = _prepared_validation_frame(baseline_chain, spot, iv_column) if baseline_chain is not None else pd.DataFrame()
    modes = [
        _validate_svi_mode(
            "Standard SVI",
            train,
            holdout,
            spot,
            baseline=baseline,
            weight_column=None,
            use_weight_fallbacks=False,
            loss="linear",
            fit_policy="unweighted_linear_loss",
        ),
        _validate_svi_mode(
            "Robust SVI",
            train,
            holdout,
            spot,
            baseline=baseline,
            weight_column="fitWeight",
            use_weight_fallbacks=True,
            loss="soft_l1",
            fit_policy="weighted_quote_reliability_soft_l1",
        ),
        _validate_ssvi_mode(
            "Robust SSVI",
            train,
            holdout,
            spot,
            baseline=baseline,
            fit_policy="weighted_global_ssvi_soft_l1",
        ),
    ]
    available_modes = [mode for mode in modes if mode.get("available")]
    return {
        "available": bool(available_modes),
        "source": "deterministic_holdout_validation",
        "provenance": VALIDATION_PROVENANCE,
        "estimate_warning": "Validation compares observed quote IV inputs with fitted estimates; fitted values are not market observations.",
        "input_rows": int(len(prepared)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(holdout)),
        "holdout_policy": f"expiry_sorted_every_{int(max(2, holdout_stride))}_row",
        "modes": modes,
        "best_oos_rmse_mode": _best_mode(available_modes, "oos_rmse"),
        "lowest_no_arb_rate_mode": _best_mode(available_modes, "no_arbitrage_violation_rate"),
    }


def backtest_fit_modes(
    observations: Sequence[MarketDataSnapshot | dict[str, Any]],
    *,
    iv_column: str = "computedIV",
) -> dict[str, Any]:
    """Compare standard, robust, and prior-assisted estimates over snapshots."""
    snapshots = [_snapshot_record(item, iv_column) for item in observations]
    snapshots = [item for item in snapshots if item["spot"] > 0 and not item["chain"].empty]
    snapshots.sort(key=lambda item: item["timestamp"])
    if len(snapshots) < 2:
        return {
            "available": False,
            "reason": "Need at least two snapshots with usable chains",
            "provenance": BACKTEST_PROVENANCE,
            "transitions": [],
        }

    transitions = []
    for previous, current in zip(snapshots[:-1], snapshots[1:]):
        transitions.append(_backtest_transition(previous, current, iv_column))

    flags = [item for item in transitions if item.get("robust_improves_stability") or item.get("hides_real_move_risk")]
    return {
        "available": bool(transitions),
        "source": "local_snapshot_sequence",
        "provenance": BACKTEST_PROVENANCE,
        "estimate_warning": "Backtest movement and stability compare fitted estimates, not market observations.",
        "snapshot_count": len(snapshots),
        "transition_count": len(transitions),
        "transitions": transitions,
        "flagged_transition_count": len(flags),
        "robust_improvement_count": int(sum(1 for item in transitions if item.get("robust_improves_stability"))),
        "hides_real_move_risk_count": int(sum(1 for item in transitions if item.get("hides_real_move_risk"))),
    }


def fixture_snapshot_record(
    label: str,
    chain: pd.DataFrame,
    spot: float,
    timestamp: datetime,
    *,
    quality_score: float | None = None,
    reason_buckets: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a lightweight snapshot record for deterministic tests/scripts."""
    return {
        "label": label,
        "symbol": "AAPL",
        "spot": float(spot),
        "timestamp": timestamp,
        "chain": chain.copy(),
        "quality_score": quality_score,
        "reason_buckets": dict(reason_buckets or {}),
    }


def _validate_svi_mode(
    mode: str,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    spot: float,
    *,
    baseline: pd.DataFrame,
    weight_column: str | None,
    use_weight_fallbacks: bool,
    loss: str,
    fit_policy: str,
) -> dict[str, Any]:
    fitted = calibrate_svi_by_expiry(
        train,
        spot,
        iv_column="validationIV",
        weight_column=weight_column,
        use_weight_fallbacks=use_weight_fallbacks,
        loss=loss,
    )
    estimates = _svi_estimates(holdout, fitted, spot)
    if estimates.empty:
        return _unavailable_mode(mode, fit_policy, "No holdout rows matched fitted expiries")
    if baseline.empty:
        baseline_estimates = pd.DataFrame()
    else:
        baseline_fitted = calibrate_svi_by_expiry(
            baseline,
            spot,
            iv_column="validationIV",
            weight_column=weight_column,
            use_weight_fallbacks=use_weight_fallbacks,
            loss=loss,
        )
        baseline_estimates = _svi_estimates(baseline, baseline_fitted, spot)
    return _mode_metrics(mode, fit_policy, estimates, spot, baseline_estimates=baseline_estimates)


def _validate_ssvi_mode(
    mode: str,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    spot: float,
    *,
    baseline: pd.DataFrame,
    fit_policy: str,
) -> dict[str, Any]:
    fitted = calibrate_ssvi_surface(train, spot, iv_column="validationIV")
    estimates = _ssvi_estimates(holdout, fitted, spot)
    if estimates.empty:
        return _unavailable_mode(mode, fit_policy, fitted.get("reason") or "No holdout rows matched fitted surface")
    baseline_fitted = calibrate_ssvi_surface(baseline, spot, iv_column="validationIV") if not baseline.empty else {}
    baseline_estimates = _ssvi_estimates(baseline, baseline_fitted, spot) if baseline_fitted else pd.DataFrame()
    return _mode_metrics(mode, fit_policy, estimates, spot, baseline_estimates=baseline_estimates)


def _mode_metrics(
    mode: str,
    fit_policy: str,
    estimates: pd.DataFrame,
    spot: float,
    *,
    baseline_estimates: pd.DataFrame,
) -> dict[str, Any]:
    residuals = estimates["validation_residual"].to_numpy(dtype=float)
    abs_residuals = np.abs(residuals)
    expiry = _expiry_residuals(estimates)
    no_arb = _estimate_no_arb(estimates, spot, mode)
    stability = _stability_metrics(estimates, baseline_estimates)
    smoothness = _smoothness_penalty(estimates)
    cell_count = int(no_arb.get("cell_count") or 0)
    violation_count = int(no_arb.get("violation_count") or 0)
    return {
        "mode": mode,
        "available": True,
        "status": "validated",
        "fit_policy": fit_policy,
        "estimate_type": "fitted_surface_estimate",
        "provenance": VALIDATION_PROVENANCE,
        "points": int(len(estimates)),
        "oos_rmse": float(np.sqrt(np.mean(residuals**2))),
        "oos_mae": float(np.mean(abs_residuals)),
        "out_of_sample_residuals_by_expiry": expiry,
        "residual_quantiles": _quantiles(abs_residuals),
        "stability_vs_prior_day": stability,
        "no_arbitrage_violation_rate": float(violation_count / cell_count) if cell_count else None,
        "no_arbitrage_violation_count": violation_count,
        "no_arbitrage_cell_count": cell_count,
        "smoothness_penalty": smoothness,
    }


def _backtest_transition(previous: dict[str, Any], current: dict[str, Any], iv_column: str) -> dict[str, Any]:
    validation = validate_fit_modes(
        current["chain"],
        current["spot"],
        iv_column=iv_column,
        baseline_chain=previous["chain"],
    )
    mode_rows = {row["mode"]: row for row in validation.get("modes", []) if row.get("available")}
    standard = mode_rows.get("Standard SVI", {})
    robust = mode_rows.get("Robust SVI", {})
    prior = _prior_assisted_transition(previous, current)
    quality_change = _quality_change(previous, current)
    standard_move = _metric_value(standard.get("stability_vs_prior_day"), "mean_abs_estimate_change")
    robust_move = _metric_value(robust.get("stability_vs_prior_day"), "mean_abs_estimate_change")
    deteriorated = quality_change["quality_deteriorated"] or bool(quality_change["deteriorated_buckets"])
    robust_improves = bool(
        standard_move is not None
        and robust_move is not None
        and robust_move < standard_move * 0.95
        and deteriorated
    )
    hides_real_move = bool(
        standard_move is not None
        and robust_move is not None
        and standard_move > 0.015
        and robust_move < standard_move * 0.5
        and not deteriorated
    )
    return {
        "from": previous["label"],
        "to": current["label"],
        "previous_timestamp": previous["timestamp"].isoformat(),
        "current_timestamp": current["timestamp"].isoformat(),
        "validation": validation,
        "prior_assisted": prior,
        "quality_change": quality_change,
        "robust_improves_stability": robust_improves,
        "hides_real_move_risk": hides_real_move or bool(prior.get("hides_real_move_risk")),
        "interpretation": _transition_interpretation(robust_improves, hides_real_move or bool(prior.get("hides_real_move_risk"))),
    }


def _prior_assisted_transition(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_estimates = _mode_point_estimates(previous["chain"], previous["spot"], robust=True)
    current_estimates = _mode_point_estimates(current["chain"], current["spot"], robust=True)
    if previous_estimates.empty or current_estimates.empty:
        return {"available": False, "reason": "Robust fitted estimates unavailable", "mode": "Prior Assisted"}
    joined = current_estimates.merge(
        previous_estimates,
        on=["expiration_key", "strike_key"],
        suffixes=("_current", "_previous"),
    )
    if joined.empty:
        return {"available": False, "reason": "No common anchors for prior-assisted transition", "mode": "Prior Assisted"}
    quality = current.get("quality_score")
    quality_value = float(quality) if quality is not None and np.isfinite(float(quality)) else 100.0
    weight = float(np.clip((80.0 - quality_value) / 80.0, 0.0, 1.0) * 0.35)
    joined["prior_assisted_iv"] = (1.0 - weight) * joined["fitted_iv_current"] + weight * joined["fitted_iv_previous"]
    joined["robust_change"] = joined["fitted_iv_current"] - joined["fitted_iv_previous"]
    joined["prior_assisted_change"] = joined["prior_assisted_iv"] - joined["fitted_iv_previous"]
    robust_mean = float(joined["robust_change"].abs().mean())
    assisted_mean = float(joined["prior_assisted_change"].abs().mean())
    deteriorated = _quality_change(previous, current)["quality_deteriorated"]
    hides_real_move = bool(weight > 0.0 and robust_mean > 0.015 and assisted_mean < robust_mean * 0.6 and not deteriorated)
    return {
        "available": True,
        "mode": "Prior Assisted",
        "estimate_type": "prior_assisted_fit_estimate",
        "provenance": PRIOR_ASSISTED_PROVENANCE,
        "blend_weight": weight,
        "matched_points": int(len(joined)),
        "robust_mean_abs_estimate_change": robust_mean,
        "prior_assisted_mean_abs_estimate_change": assisted_mean,
        "hides_real_move_risk": hides_real_move,
    }


def _prepared_validation_frame(chain: pd.DataFrame | None, spot: float, iv_column: str) -> pd.DataFrame:
    if chain is None or chain.empty or spot <= 0:
        return pd.DataFrame()
    work = chain.copy()
    if iv_column not in work:
        iv_column = "impliedVolatility"
    required = {"expiration", "strike", "daysToExpiration", iv_column}
    if not required.issubset(work.columns):
        return pd.DataFrame()
    work["expiration_norm"] = pd.to_datetime(work["expiration"], errors="coerce").dt.normalize()
    work["expiration_key"] = work["expiration_norm"].dt.date.astype(str)
    work["strike_num"] = pd.to_numeric(work["strike"], errors="coerce")
    work["strike_key"] = work["strike_num"].round(8)
    work["dte_num"] = pd.to_numeric(work["daysToExpiration"], errors="coerce")
    work["validationIV"] = pd.to_numeric(work[iv_column], errors="coerce")
    if "type" in work:
        work["type_key"] = work["type"].astype(str).str.lower()
    else:
        work["type_key"] = "unknown"
    work = work.dropna(subset=["expiration_norm", "strike_num", "dte_num", "validationIV"])
    work = work[(work["strike_num"] > 0.0) & (work["dte_num"] > 0.0) & (work["validationIV"] > 0.0)].copy()
    work["logMoneyness"] = np.log(work["strike_num"] / float(spot))
    return work


def _train_holdout_split(frame: pd.DataFrame, holdout_stride: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stride = max(2, int(holdout_stride))
    ordered = frame.sort_values(["expiration_key", "strike_num", "type_key"]).copy()
    parts = []
    for _, group in ordered.groupby("expiration_key", sort=True):
        local = group.copy()
        local["_validation_rank"] = np.arange(len(local))
        parts.append(local)
    ranked = pd.concat(parts, ignore_index=True) if parts else ordered
    mask = ranked["_validation_rank"] % stride == stride - 1
    holdout = ranked[mask].drop(columns=["_validation_rank"]).copy()
    train = ranked[~mask].drop(columns=["_validation_rank"]).copy()
    return train, holdout


def _svi_estimates(frame: pd.DataFrame, fitted: pd.DataFrame, spot: float) -> pd.DataFrame:
    if frame.empty or fitted.empty:
        return pd.DataFrame()
    params = {str(row["expiration"]): row for row in fitted.to_dict("records")}
    rows = []
    for row in frame.to_dict("records"):
        fit = params.get(str(row.get("expiration_key")))
        if not fit:
            continue
        t = max(float(row["dte_num"]) / 365.0, 1e-8)
        total_variance = svi_total_variance(
            np.array([float(row["logMoneyness"])]),
            float(fit["a"]),
            float(fit["b"]),
            float(fit["rho"]),
            float(fit["m"]),
            float(fit["sigma"]),
        )[0]
        fitted_iv = float(np.sqrt(max(total_variance, 1e-10) / t))
        rows.append(_estimate_row(row, fitted_iv, spot))
    return pd.DataFrame(rows)


def _ssvi_estimates(frame: pd.DataFrame, fitted: dict[str, Any], spot: float) -> pd.DataFrame:
    if frame.empty or fitted.get("status") != "fitted":
        return pd.DataFrame()
    theta = {str(row["expiration"]): float(row["theta"]) for row in fitted.get("atm_total_variance") or []}
    rows = []
    for row in frame.to_dict("records"):
        expiry = str(row.get("expiration_key"))
        if expiry not in theta:
            continue
        t = max(float(row["dte_num"]) / 365.0, 1e-8)
        total_variance = ssvi_total_variance(
            np.array([float(row["logMoneyness"])]),
            np.array([theta[expiry]]),
            float(fitted["rho"]),
            float(fitted["eta"]),
            float(fitted["gamma"]),
        )[0]
        fitted_iv = float(np.sqrt(max(total_variance, 1e-10) / t))
        rows.append(_estimate_row(row, fitted_iv, spot))
    return pd.DataFrame(rows)


def _estimate_row(source: dict[str, Any], fitted_iv: float, spot: float) -> dict[str, Any]:
    observed = float(source["validationIV"])
    return {
        "expiration_key": str(source["expiration_key"]),
        "dte": float(source["dte_num"]),
        "strike": float(source["strike_num"]),
        "strike_key": float(source["strike_key"]),
        "type_key": str(source.get("type_key") or "unknown"),
        "log_moneyness": float(np.log(float(source["strike_num"]) / float(spot))),
        "observed_iv": observed,
        "fitted_iv": fitted_iv,
        "validation_residual": fitted_iv - observed,
    }


def _mode_point_estimates(chain: pd.DataFrame, spot: float, *, robust: bool) -> pd.DataFrame:
    prepared = _prepared_validation_frame(chain, spot, "computedIV")
    if prepared.empty:
        return pd.DataFrame()
    fitted = calibrate_svi_by_expiry(
        prepared,
        spot,
        iv_column="validationIV",
        weight_column="fitWeight" if robust else None,
        use_weight_fallbacks=robust,
        loss="soft_l1" if robust else "linear",
    )
    return _svi_estimates(prepared, fitted, spot)


def _expiry_residuals(estimates: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for expiry, group in estimates.groupby("expiration_key", sort=True):
        residuals = group["validation_residual"].to_numpy(dtype=float)
        rows.append(
            {
                "expiration": str(expiry),
                "points": int(len(group)),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "mae": float(np.mean(np.abs(residuals))),
                "mean_residual": float(np.mean(residuals)),
                "p95_abs_residual": float(pd.Series(np.abs(residuals)).quantile(0.95)),
            }
        )
    return rows


def _estimate_no_arb(estimates: pd.DataFrame, spot: float, mode: str) -> dict[str, Any]:
    grid = _estimate_grid(estimates)
    if grid is None:
        return {"violation_count": 0, "cell_count": 0}
    strikes, dtes, vols = grid
    check = check_surface_arbitrage(strikes, dtes, vols, spot, surface_label=f"{mode} validation estimate")
    return {
        "violation_count": int(check.get("violation_count") or 0),
        "cell_count": int(np.asarray(vols).size),
        "reason_buckets": check.get("reason_buckets") or {},
    }


def _estimate_grid(estimates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if estimates.empty:
        return None
    grouped = (
        estimates.groupby(["dte", "strike"], as_index=False)["fitted_iv"]
        .mean()
        .sort_values(["dte", "strike"])
    )
    pivot = grouped.pivot(index="dte", columns="strike", values="fitted_iv")
    if pivot.empty:
        return None
    pivot = pivot.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if pivot.empty:
        return None
    filled = pivot.interpolate(axis=1, limit_direction="both").interpolate(axis=0, limit_direction="both")
    return (
        filled.columns.to_numpy(dtype=float),
        filled.index.to_numpy(dtype=float),
        filled.to_numpy(dtype=float),
    )


def _smoothness_penalty(estimates: pd.DataFrame) -> dict[str, Any]:
    penalties = []
    max_change = 0.0
    for _, group in estimates.sort_values(["expiration_key", "strike"]).groupby("expiration_key"):
        values = group["fitted_iv"].to_numpy(dtype=float)
        if len(values) < 2:
            continue
        diffs = np.diff(values)
        penalties.extend(np.square(diffs).tolist())
        max_change = max(max_change, float(np.max(np.abs(diffs))))
    return {
        "mean_squared_adjacent_iv_change": float(np.mean(penalties)) if penalties else None,
        "max_adjacent_iv_change": max_change if penalties else None,
        "adjacent_pair_count": len(penalties),
    }


def _stability_metrics(estimates: pd.DataFrame, baseline_estimates: pd.DataFrame) -> dict[str, Any]:
    if estimates.empty or baseline_estimates.empty:
        return {
            "available": False,
            "reason": "No baseline fitted estimates supplied",
            "provenance": VALIDATION_PROVENANCE,
        }
    matched = estimates.merge(
        baseline_estimates[["expiration_key", "strike_key", "type_key", "fitted_iv"]],
        on=["expiration_key", "strike_key", "type_key"],
        suffixes=("_current", "_baseline"),
    )
    if matched.empty:
        return {
            "available": False,
            "reason": "No common validation anchors with baseline",
            "provenance": VALIDATION_PROVENANCE,
        }
    matched["estimate_change"] = matched["fitted_iv_current"] - matched["fitted_iv_baseline"]
    return {
        "available": True,
        "provenance": VALIDATION_PROVENANCE,
        "matched_points": int(len(matched)),
        "mean_estimate_change": float(matched["estimate_change"].mean()),
        "mean_abs_estimate_change": float(matched["estimate_change"].abs().mean()),
        "max_abs_estimate_change": float(matched["estimate_change"].abs().max()),
    }


def _snapshot_record(item: MarketDataSnapshot | dict[str, Any], iv_column: str) -> dict[str, Any]:
    if isinstance(item, MarketDataSnapshot):
        return {
            "label": item.spot_timestamp.isoformat(),
            "symbol": item.symbol,
            "spot": float(item.spot),
            "timestamp": item.spot_timestamp,
            "chain": item.options_frame(),
            "quality_score": item.quality_score or item.data_quality_score,
            "reason_buckets": dict(item.quality_reason_buckets or item.rejection_reasons),
        }
    timestamp = pd.to_datetime(item.get("timestamp"), errors="coerce")
    return {
        "label": str(item.get("label") or (timestamp.isoformat() if pd.notna(timestamp) else "snapshot")),
        "symbol": str(item.get("symbol") or "UNKNOWN"),
        "spot": float(item.get("spot") or 0.0),
        "timestamp": timestamp.to_pydatetime() if pd.notna(timestamp) else datetime.min,
        "chain": item.get("chain", pd.DataFrame()).copy(),
        "quality_score": item.get("quality_score"),
        "reason_buckets": dict(item.get("reason_buckets") or {}),
        "iv_column": iv_column,
    }


def _quality_change(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_score = previous.get("quality_score")
    current_score = current.get("quality_score")
    score_change = None
    quality_deteriorated = False
    if previous_score is not None and current_score is not None:
        score_change = float(current_score) - float(previous_score)
        quality_deteriorated = score_change <= -5.0
    previous_buckets = dict(previous.get("reason_buckets") or {})
    current_buckets = dict(current.get("reason_buckets") or {})
    deteriorated_buckets = {
        key: int(current_buckets.get(key, 0) - previous_buckets.get(key, 0))
        for key in sorted(set(current_buckets) | set(previous_buckets))
        if int(current_buckets.get(key, 0) - previous_buckets.get(key, 0)) > 0
    }
    return {
        "previous_quality_score": previous_score,
        "current_quality_score": current_score,
        "score_change": score_change,
        "quality_deteriorated": quality_deteriorated,
        "deteriorated_buckets": deteriorated_buckets,
    }


def _transition_interpretation(robust_improves: bool, hides_real_move: bool) -> str:
    if hides_real_move:
        return "Review: robust or prior-assisted estimates may be damping a real surface move."
    if robust_improves:
        return "Robust fit improved stability while data-quality buckets deteriorated."
    return "No stability warning triggered."


def _metric_value(payload: Any, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return float(value) if value is not None and np.isfinite(float(value)) else None


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    clean = pd.Series(values, dtype="float64").dropna()
    if clean.empty:
        return {"p50": None, "p90": None, "p95": None, "p99": None}
    return {name: float(clean.quantile(q)) for name, q in {"p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99}.items()}


def _best_mode(modes: list[dict[str, Any]], metric: str) -> str | None:
    ranked = [mode for mode in modes if mode.get(metric) is not None]
    if not ranked:
        return None
    return str(min(ranked, key=lambda row: float(row[metric]))["mode"])


def _unavailable_validation(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "source": "deterministic_holdout_validation",
        "provenance": VALIDATION_PROVENANCE,
        "modes": [],
    }


def _unavailable_mode(mode: str, fit_policy: str, reason: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "available": False,
        "status": "unavailable",
        "fit_policy": fit_policy,
        "reason": reason,
        "provenance": VALIDATION_PROVENANCE,
    }
