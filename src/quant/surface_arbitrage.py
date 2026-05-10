"""Post-fit arbitrage diagnostics and opt-in surface repair.

These checks operate on fitted IV estimates, not observed market quotes. Any
repair output is explicitly labeled as an estimate so downstream views can keep
raw observations, fitted estimates, prior-assisted estimates, and repaired
candidates distinct.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


DIAGNOSTIC_PROVENANCE = "post_fit_surface_arbitrage_diagnostic_not_market_observation"
REPAIR_PROVENANCE = "conservative_surface_repair_estimate_not_market_observation"


def check_surface_arbitrage(
    strikes: Any,
    expiries: Any,
    vols: Any,
    spot: float,
    *,
    input_rows: pd.DataFrame | None = None,
    surface_label: str = "surface_estimate",
    min_vol: float = 0.01,
    max_vol: float = 5.0,
    calendar_tolerance: float = 1e-8,
    convexity_tolerance: float = 1e-7,
    smoothness_limit: float = 0.18,
    max_records: int = 50,
) -> dict[str, Any]:
    """Check positive vols, calendar monotonicity, convexity, and smoothness."""
    cells = _surface_cells(strikes, expiries, vols, spot)
    if cells.empty:
        return _empty_check(surface_label, "Surface grid unavailable")

    violations: list[dict[str, Any]] = []
    _positive_vol_violations(cells, min_vol, max_vol, violations)
    _calendar_violations(cells, calendar_tolerance, violations)
    _butterfly_convexity_violations(cells, convexity_tolerance, violations)
    _smoothness_violations(cells, smoothness_limit, violations)

    suggestions = _repair_suggestions(violations, input_rows, spot, max_records=max_records)
    reason_buckets = Counter(str(item["check"]) for item in violations)
    metrics = _surface_metrics(cells, smoothness_limit)
    metrics.update(
        {
            "positive_vol_violations": int(reason_buckets.get("positive_vol", 0)),
            "calendar_violations": int(reason_buckets.get("calendar_monotonicity", 0)),
            "butterfly_convexity_violations": int(reason_buckets.get("butterfly_convexity", 0)),
            "smoothness_violations": int(reason_buckets.get("smoothness_bound", 0)),
        }
    )
    return {
        "surface_label": surface_label,
        "checks": [
            "positive_finite_vols",
            "calendar_total_variance_monotonicity",
            "butterfly_convexity_total_variance",
            "smoothness_bounds",
        ],
        "passed": not violations,
        "violation_count": int(len(violations)),
        "reason_buckets": dict(sorted((key, int(value)) for key, value in reason_buckets.items())),
        "violations": [_json_safe(row) for row in violations[:max_records]],
        "suggestions": suggestions,
        "metrics": metrics,
        "provenance": DIAGNOSTIC_PROVENANCE,
        "estimate_warning": "Surface arbitrage diagnostics evaluate fitted estimates, not market observations.",
    }


def conservative_surface_repair(
    strikes: Any,
    expiries: Any,
    vols: Any,
    spot: float,
    *,
    enabled: bool = False,
    min_vol: float = 0.01,
    max_vol: float = 5.0,
    max_iterations: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an opt-in conservative repaired IV grid and provenance metadata."""
    original = np.asarray(vols, dtype=float)
    before = check_surface_arbitrage(
        strikes,
        expiries,
        original,
        spot,
        surface_label="pre_repair_surface_estimate",
        min_vol=min_vol,
        max_vol=max_vol,
    )
    if not enabled or original.size == 0:
        return original.copy(), {
            "enabled": bool(enabled),
            "applied": False,
            "reason": "Conservative repair disabled" if not enabled else "Surface grid unavailable",
            "before": before,
            "after": before,
            "repaired_cell_count": 0,
            "repair_records": [],
            "provenance": REPAIR_PROVENANCE,
            "estimate_warning": "Repair candidates are estimates and are not market observations.",
        }

    cells = _surface_cells(strikes, expiries, original, spot)
    if cells.empty:
        return original.copy(), {
            "enabled": True,
            "applied": False,
            "reason": "Surface grid unavailable",
            "before": before,
            "after": before,
            "repaired_cell_count": 0,
            "repair_records": [],
            "provenance": REPAIR_PROVENANCE,
            "estimate_warning": "Repair candidates are estimates and are not market observations.",
        }

    repaired = original.copy().reshape(-1)
    cells["repaired_iv"] = np.clip(cells["iv"].to_numpy(dtype=float), min_vol, max_vol)
    cells["repaired_iv"] = cells["repaired_iv"].where(np.isfinite(cells["repaired_iv"]), min_vol)
    for _ in range(max(1, int(max_iterations))):
        _project_calendar(cells, min_vol=min_vol, max_vol=max_vol)
        _project_convexity(cells, min_vol=min_vol, max_vol=max_vol)
    for row in cells.itertuples():
        repaired[int(row.flat_index)] = float(row.repaired_iv)
    repaired_grid = repaired.reshape(original.shape)
    after = check_surface_arbitrage(
        strikes,
        expiries,
        repaired_grid,
        spot,
        surface_label="conservative_repair_candidate",
        min_vol=min_vol,
        max_vol=max_vol,
    )
    repair_records = _repair_records(cells)
    return repaired_grid, {
        "enabled": True,
        "applied": bool(repair_records),
        "reason": None if repair_records else "No repair adjustments required",
        "before": before,
        "after": after,
        "before_violation_count": before["violation_count"],
        "after_violation_count": after["violation_count"],
        "violation_reduction": int(before["violation_count"] - after["violation_count"]),
        "repaired_cell_count": int(len(repair_records)),
        "repair_records": repair_records,
        "provenance": REPAIR_PROVENANCE,
        "estimate_warning": "Repair candidates are estimates and are not market observations.",
    }


def surface_comparison_rows(
    strikes: Any,
    expiries: Any,
    spot: float,
    *,
    current_vols: Any,
    prior_assisted_vols: Any | None = None,
    repaired_vols: Any | None = None,
    prior_metadata: dict[str, Any] | None = None,
    repair_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic comparison rows for fitted and derived surfaces."""
    rows = [
        _comparison_row(
            "Robust Surface",
            "current_robust_fit_estimate",
            check_surface_arbitrage(strikes, expiries, current_vols, spot, surface_label="current_robust_fit"),
        )
    ]
    if prior_assisted_vols is not None:
        prior_check = check_surface_arbitrage(
            strikes,
            expiries,
            prior_assisted_vols,
            spot,
            surface_label="prior_assisted_estimate",
        )
        rows.append(
            _comparison_row(
                "Prior Assisted",
                "prior_assisted_estimate",
                prior_check,
                prior_weight=(prior_metadata or {}).get("blend_weight"),
                status="applied" if (prior_metadata or {}).get("applied") else "not_applied",
            )
        )
    if repaired_vols is not None:
        after = (repair_metadata or {}).get("after") or check_surface_arbitrage(
            strikes,
            expiries,
            repaired_vols,
            spot,
            surface_label="conservative_repair_candidate",
        )
        rows.append(
            _comparison_row(
                "Conservative Repair",
                "conservative_repair_candidate_not_applied",
                after,
                repair_applied=bool((repair_metadata or {}).get("applied")),
                status="candidate",
            )
        )
    return rows


def _comparison_row(
    mode: str,
    estimate_type: str,
    check: dict[str, Any],
    *,
    prior_weight: Any = None,
    repair_applied: bool | None = None,
    status: str = "fitted",
) -> dict[str, Any]:
    metrics = check.get("metrics") or {}
    buckets = check.get("reason_buckets") or {}
    return {
        "mode": mode,
        "status": status,
        "estimate_type": estimate_type,
        "arbitrage_violations": check.get("violation_count"),
        "calendar_violations": buckets.get("calendar_monotonicity", 0),
        "butterfly_convexity_violations": buckets.get("butterfly_convexity", 0),
        "positive_vol_violations": buckets.get("positive_vol", 0),
        "smoothness_violations": buckets.get("smoothness_bound", 0),
        "surface_roughness": metrics.get("roughness"),
        "smoothness_max_adjacent_iv_change": metrics.get("max_adjacent_iv_change"),
        "prior_weight": prior_weight,
        "repair_applied": repair_applied,
        "provenance": check.get("provenance"),
    }


def _surface_cells(strikes: Any, expiries: Any, vols: Any, spot: float) -> pd.DataFrame:
    vol_grid = np.asarray(vols, dtype=float)
    if vol_grid.size == 0:
        return pd.DataFrame()
    strike_grid, expiry_grid = _surface_mesh(strikes, expiries, vol_grid.shape)
    rows = pd.DataFrame(
        {
            "flat_index": np.arange(vol_grid.size, dtype=int),
            "strike": np.asarray(strike_grid, dtype=float).reshape(-1),
            "dte": np.asarray(expiry_grid, dtype=float).reshape(-1),
            "iv": vol_grid.reshape(-1),
        }
    )
    rows = rows.replace([np.inf, -np.inf], np.nan)
    rows = rows.dropna(subset=["strike", "dte"])
    rows = rows[(rows["strike"] > 0.0) & (rows["dte"] > 0.0)].copy()
    if rows.empty:
        return rows
    rows["time"] = rows["dte"] / 365.0
    rows["log_moneyness"] = np.log(rows["strike"] / float(spot)) if spot > 0.0 else np.nan
    rows["dte_key"] = rows["dte"].round(8)
    rows["log_moneyness_key"] = rows["log_moneyness"].round(8)
    rows["total_variance"] = rows["iv"] ** 2 * rows["time"]
    return rows.sort_values(["dte", "log_moneyness", "flat_index"]).reset_index(drop=True)


def _surface_mesh(strikes: Any, expiries: Any, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    strike_values = np.asarray(strikes, dtype=float)
    expiry_values = np.asarray(expiries, dtype=float)
    if strike_values.shape == shape and expiry_values.shape == shape:
        return strike_values, expiry_values
    if len(shape) == 2 and strike_values.ndim == 1 and expiry_values.ndim == 1:
        if shape == (len(expiry_values), len(strike_values)):
            return np.meshgrid(strike_values, expiry_values)
        if shape == (len(strike_values), len(expiry_values)):
            expiry_grid, strike_grid = np.meshgrid(expiry_values, strike_values)
            return strike_grid, expiry_grid
    if len(shape) == 2 and strike_values.shape == shape and expiry_values.ndim == 1:
        return strike_values, np.repeat(expiry_values.reshape(-1, 1), shape[1], axis=1)
    if len(shape) == 2 and expiry_values.shape == shape and strike_values.ndim == 1:
        return np.repeat(strike_values.reshape(1, -1), shape[0], axis=0), expiry_values
    return np.broadcast_to(strike_values, shape), np.broadcast_to(expiry_values, shape)


def _positive_vol_violations(cells: pd.DataFrame, min_vol: float, max_vol: float, violations: list[dict[str, Any]]) -> None:
    bad = cells[(~np.isfinite(cells["iv"])) | (cells["iv"] < min_vol) | (cells["iv"] > max_vol)]
    for row in bad.itertuples():
        violations.append(
            {
                "check": "positive_vol",
                "dte": float(row.dte),
                "strike": float(row.strike),
                "log_moneyness": float(row.log_moneyness),
                "iv": _none_if_nan(row.iv),
                "suggestion": "Clip fitted IV into configured positive bounds before downstream use.",
            }
        )


def _calendar_violations(cells: pd.DataFrame, tolerance: float, violations: list[dict[str, Any]]) -> None:
    valid = cells.dropna(subset=["iv", "total_variance", "log_moneyness_key"])
    valid = valid[(valid["iv"] > 0.0) & (valid["time"] > 0.0)]
    for _, group in valid.groupby("log_moneyness_key", sort=True):
        if len(group) < 2:
            continue
        group = group.sort_values("dte")
        for front, back in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            if float(back.total_variance) + tolerance >= float(front.total_variance):
                continue
            violations.append(
                {
                    "check": "calendar_monotonicity",
                    "front_dte": float(front.dte),
                    "back_dte": float(back.dte),
                    "strike": float(back.strike),
                    "log_moneyness": float(back.log_moneyness),
                    "front_total_variance": float(front.total_variance),
                    "back_total_variance": float(back.total_variance),
                    "suggestion": "Raise the later total variance minimally or inspect stale front-expiry inputs.",
                }
            )


def _butterfly_convexity_violations(cells: pd.DataFrame, tolerance: float, violations: list[dict[str, Any]]) -> None:
    valid = cells.dropna(subset=["iv", "total_variance", "dte_key", "log_moneyness"])
    valid = valid[(valid["iv"] > 0.0) & (valid["time"] > 0.0)]
    for _, group in valid.groupby("dte_key", sort=True):
        if len(group) < 3:
            continue
        group = group.sort_values("log_moneyness")
        rows = list(group.itertuples())
        for left, middle, right in zip(rows[:-2], rows[1:-1], rows[2:]):
            span = float(right.log_moneyness - left.log_moneyness)
            if span <= 0.0:
                continue
            left_weight = float((right.log_moneyness - middle.log_moneyness) / span)
            right_weight = 1.0 - left_weight
            chord = left_weight * float(left.total_variance) + right_weight * float(right.total_variance)
            if float(middle.total_variance) <= chord + tolerance:
                continue
            violations.append(
                {
                    "check": "butterfly_convexity",
                    "dte": float(middle.dte),
                    "strike": float(middle.strike),
                    "log_moneyness": float(middle.log_moneyness),
                    "total_variance": float(middle.total_variance),
                    "convex_upper": float(chord),
                    "suggestion": "Lower the local total-variance peak or inspect nearby low-reliability quote rows.",
                }
            )


def _smoothness_violations(cells: pd.DataFrame, limit: float, violations: list[dict[str, Any]]) -> None:
    valid = cells.dropna(subset=["iv", "dte_key", "log_moneyness_key"])
    for _, group in valid.groupby("dte_key", sort=True):
        group = group.sort_values("log_moneyness")
        for left, right in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            change = abs(float(right.iv) - float(left.iv))
            if change <= limit:
                continue
            violations.append(
                {
                    "check": "smoothness_bound",
                    "dte": float(right.dte),
                    "strike": float(right.strike),
                    "log_moneyness": float(right.log_moneyness),
                    "adjacent_iv_change": float(change),
                    "limit": float(limit),
                    "suggestion": "Inspect adjacent strike inputs or enable conservative smoothing in repair mode.",
                }
            )
    for _, group in valid.groupby("log_moneyness_key", sort=True):
        group = group.sort_values("dte")
        for front, back in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            change = abs(float(back.iv) - float(front.iv))
            if change <= limit:
                continue
            violations.append(
                {
                    "check": "smoothness_bound",
                    "front_dte": float(front.dte),
                    "back_dte": float(back.dte),
                    "strike": float(back.strike),
                    "log_moneyness": float(back.log_moneyness),
                    "adjacent_iv_change": float(change),
                    "limit": float(limit),
                    "suggestion": "Inspect adjacent expiry inputs or enable conservative smoothing in repair mode.",
                }
            )


def _repair_suggestions(
    violations: list[dict[str, Any]],
    input_rows: pd.DataFrame | None,
    spot: float,
    *,
    max_records: int,
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for violation in violations[:max_records]:
        dte = violation.get("dte") or violation.get("back_dte") or violation.get("front_dte")
        log_money = violation.get("log_moneyness")
        suggestions.append(
            {
                "check": violation.get("check"),
                "location": {
                    "dte": dte,
                    "strike": violation.get("strike"),
                    "log_moneyness": log_money,
                },
                "suggestion": violation.get("suggestion"),
                "likely_input_rows": _likely_input_rows(input_rows, spot, dte, log_money),
            }
        )
    return suggestions


def _likely_input_rows(
    input_rows: pd.DataFrame | None,
    spot: float,
    dte: Any,
    log_money: Any,
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    if input_rows is None or input_rows.empty or dte is None or log_money is None:
        return []
    work = pd.DataFrame(index=input_rows.index)
    work["dte"] = pd.to_numeric(input_rows.get("daysToExpiration"), errors="coerce")
    if "logMoneyness" in input_rows:
        work["log_moneyness"] = pd.to_numeric(input_rows["logMoneyness"], errors="coerce")
    else:
        strikes = pd.to_numeric(input_rows.get("strike"), errors="coerce")
        work["log_moneyness"] = np.log(strikes / float(spot)) if spot > 0.0 else np.nan
    work["distance"] = ((work["dte"] - float(dte)) / 30.0) ** 2 + (work["log_moneyness"] - float(log_money)) ** 2
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["distance"]).sort_values("distance").head(top_n)
    rows = []
    for idx in work.index:
        source = input_rows.loc[idx]
        rows.append(
            {
                "contract": _str_or_none(source.get("contractSymbol") or source.get("contract")),
                "type": _str_or_none(source.get("type")),
                "strike": _float_or_none(source.get("strike")),
                "dte": _float_or_none(source.get("daysToExpiration")),
                "iv": _float_or_none(source.get("computedIV") if "computedIV" in source else source.get("impliedVolatility")),
                "fit_weight": _float_or_none(source.get("fitWeight")),
                "quote_reliability_score": _float_or_none(source.get("quoteReliabilityScore")),
                "no_arbitrage_reasons": _str_or_none(source.get("noArbitrageReasons")),
            }
        )
    return rows


def _project_calendar(cells: pd.DataFrame, *, min_vol: float, max_vol: float) -> None:
    for _, group in cells.groupby("log_moneyness_key", sort=True):
        if len(group) < 2:
            continue
        running = 0.0
        for idx, row in group.sort_values("dte").iterrows():
            time = max(float(row["time"]), 1e-8)
            total_variance = float(row["repaired_iv"]) ** 2 * time
            running = max(running, total_variance)
            cells.loc[idx, "repaired_iv"] = float(np.clip(np.sqrt(running / time), min_vol, max_vol))


def _project_convexity(cells: pd.DataFrame, *, min_vol: float, max_vol: float) -> None:
    cells["repaired_total_variance"] = cells["repaired_iv"] ** 2 * cells["time"]
    for _, group in cells.groupby("dte_key", sort=True):
        if len(group) < 3:
            continue
        ordered = group.sort_values("log_moneyness")
        rows = list(ordered.itertuples())
        for left, middle, right in zip(rows[:-2], rows[1:-1], rows[2:]):
            span = float(right.log_moneyness - left.log_moneyness)
            if span <= 0.0:
                continue
            left_weight = float((right.log_moneyness - middle.log_moneyness) / span)
            right_weight = 1.0 - left_weight
            chord = left_weight * float(left.repaired_total_variance) + right_weight * float(right.repaired_total_variance)
            if float(middle.repaired_total_variance) <= chord:
                continue
            repaired_iv = np.sqrt(max(chord, 1e-12) / max(float(middle.time), 1e-8))
            cells.loc[middle.Index, "repaired_iv"] = float(np.clip(repaired_iv, min_vol, max_vol))
            cells.loc[middle.Index, "repaired_total_variance"] = cells.loc[middle.Index, "repaired_iv"] ** 2 * float(
                middle.time
            )


def _repair_records(cells: pd.DataFrame) -> list[dict[str, Any]]:
    changed = cells[np.abs(cells["repaired_iv"] - cells["iv"]) > 1e-10]
    records = []
    for row in changed.sort_values(["dte", "log_moneyness"]).itertuples():
        records.append(
            {
                "strike": float(row.strike),
                "dte": float(row.dte),
                "log_moneyness": float(row.log_moneyness),
                "original_iv": _none_if_nan(row.iv),
                "repaired_iv": float(row.repaired_iv),
                "iv_change": float(row.repaired_iv - row.iv) if np.isfinite(row.iv) else None,
                "provenance": REPAIR_PROVENANCE,
            }
        )
    return records


def _surface_metrics(cells: pd.DataFrame, smoothness_limit: float) -> dict[str, Any]:
    iv = pd.to_numeric(cells["iv"], errors="coerce")
    finite = iv[np.isfinite(iv)]
    max_adjacent = _max_adjacent_iv_change(cells)
    return {
        "cell_count": int(len(cells)),
        "finite_cell_count": int(len(finite)),
        "min_iv": float(finite.min()) if not finite.empty else None,
        "max_iv": float(finite.max()) if not finite.empty else None,
        "roughness": _roughness(cells),
        "max_adjacent_iv_change": max_adjacent,
        "smoothness_limit": float(smoothness_limit),
    }


def _max_adjacent_iv_change(cells: pd.DataFrame) -> float | None:
    changes: list[float] = []
    valid = cells.dropna(subset=["iv", "dte_key", "log_moneyness_key"])
    for _, group in valid.groupby("dte_key", sort=True):
        ordered = group.sort_values("log_moneyness")
        if len(ordered) > 1:
            changes.extend(np.abs(np.diff(ordered["iv"].to_numpy(dtype=float))).tolist())
    for _, group in valid.groupby("log_moneyness_key", sort=True):
        ordered = group.sort_values("dte")
        if len(ordered) > 1:
            changes.extend(np.abs(np.diff(ordered["iv"].to_numpy(dtype=float))).tolist())
    clean = [float(value) for value in changes if np.isfinite(value)]
    return max(clean) if clean else None


def _roughness(cells: pd.DataFrame) -> float:
    changes: list[float] = []
    valid = cells.dropna(subset=["iv", "dte_key", "log_moneyness_key"])
    for _, group in valid.groupby("dte_key", sort=True):
        ordered = group.sort_values("log_moneyness")
        if len(ordered) > 1:
            changes.extend((np.diff(ordered["iv"].to_numpy(dtype=float)) ** 2).tolist())
    for _, group in valid.groupby("log_moneyness_key", sort=True):
        ordered = group.sort_values("dte")
        if len(ordered) > 1:
            changes.extend((np.diff(ordered["iv"].to_numpy(dtype=float)) ** 2).tolist())
    clean = [float(value) for value in changes if np.isfinite(value)]
    return float(np.mean(clean)) if clean else 0.0


def _empty_check(surface_label: str, reason: str) -> dict[str, Any]:
    return {
        "surface_label": surface_label,
        "checks": [],
        "passed": False,
        "reason": reason,
        "violation_count": 0,
        "reason_buckets": {},
        "violations": [],
        "suggestions": [],
        "metrics": {"cell_count": 0, "finite_cell_count": 0},
        "provenance": DIAGNOSTIC_PROVENANCE,
    }


def _json_safe(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe_value(value) for key, value in row.items()}


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _none_if_nan(value)
    return value


def _none_if_nan(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _str_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
