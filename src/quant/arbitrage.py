"""Static no-arbitrage checks for normalized option chains."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CHECK_COLUMNS = [
    "noArbitrageViolation",
    "noArbitrageReasons",
    "noArbitrageLowerBound",
    "noArbitrageUpperBound",
    "noArbitrageBoundViolation",
    "noArbitrageMonotonicityViolation",
    "noArbitrageConvexityViolation",
    "noArbitrageCalendarViolation",
]


def apply_no_arbitrage_checks(
    chain: pd.DataFrame,
    spot: float,
    price_column: str = "selectedMarketPrice",
    tolerance: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Annotate option rows with simple static no-arbitrage diagnostics.

    The checks are intentionally local and deterministic: European price bounds,
    monotonicity in strike, convexity in strike, and total-variance calendar
    monotonicity when IVs are available for matching strike/type rows.
    """
    if chain.empty:
        return _empty_frame(chain), _empty_metadata()

    df = _ensure_columns(chain.copy())
    if spot <= 0 or price_column not in df:
        return df, _empty_metadata()

    work = _prepared_frame(df, price_column)
    violations: list[dict[str, Any]] = []
    reason_buckets: dict[str, int] = {}

    _check_bounds(df, work, spot, tolerance, violations, reason_buckets)
    _check_strike_monotonicity(df, work, tolerance, violations, reason_buckets)
    _check_convexity(df, work, tolerance, violations, reason_buckets)
    _check_calendar_total_variance(df, work, violations, reason_buckets)

    rows = int(df["noArbitrageViolation"].fillna(False).astype(bool).sum())
    return df, {
        "no_arbitrage_checks": [
            "bounds_by_type",
            "call_monotonicity",
            "put_monotonicity",
            "convexity",
            "calendar_total_variance",
        ],
        "no_arbitrage_violation_count": len(violations),
        "no_arbitrage_violation_rows": rows,
        "no_arbitrage_reason_buckets": {reason: int(count) for reason, count in reason_buckets.items() if count},
        "no_arbitrage_violations": violations[:50],
    }


def _empty_frame(chain: pd.DataFrame) -> pd.DataFrame:
    return _ensure_columns(chain.copy())


def _empty_metadata() -> dict[str, Any]:
    return {
        "no_arbitrage_checks": [],
        "no_arbitrage_violation_count": 0,
        "no_arbitrage_violation_rows": 0,
        "no_arbitrage_reason_buckets": {},
        "no_arbitrage_violations": [],
    }


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["noArbitrageViolation"] = False
    df["noArbitrageReasons"] = ""
    df["noArbitrageLowerBound"] = np.nan
    df["noArbitrageUpperBound"] = np.nan
    df["noArbitrageBoundViolation"] = False
    df["noArbitrageMonotonicityViolation"] = False
    df["noArbitrageConvexityViolation"] = False
    df["noArbitrageCalendarViolation"] = False
    return df


def _prepared_frame(df: pd.DataFrame, price_column: str) -> pd.DataFrame:
    work = df.copy()
    if "type" in work:
        work["type_norm"] = work["type"].astype(str).str.lower()
    else:
        work["type_norm"] = pd.Series("", index=work.index, dtype="object")
    if "expiration" in work:
        work["expiration_norm"] = pd.to_datetime(work["expiration"], errors="coerce").dt.normalize()
    else:
        work["expiration_norm"] = pd.NaT
    work["price_num"] = pd.to_numeric(work[price_column], errors="coerce")
    work["strike_num"] = _numeric_column(work, "strike")
    work["dte_num"] = _numeric_column(work, "daysToExpiration")
    work["time_num"] = _time_to_expiry(work)
    work["rate_num"] = _numeric_column(work, "riskFreeRate").fillna(0.0)
    dividend = _numeric_column(work, "effectiveDividendYield")
    if dividend.isna().all():
        dividend = _numeric_column(work, "dividendYield")
    work["dividend_num"] = dividend.fillna(0.0)
    iv = _numeric_column(work, "computedIV")
    if iv.isna().all():
        iv = _numeric_column(work, "impliedVolatility")
    work["iv_num"] = iv
    return work


def _time_to_expiry(work: pd.DataFrame) -> pd.Series:
    time = _numeric_column(work, "time_to_expiry")
    dte_time = _numeric_column(work, "daysToExpiration") / 365.0
    return time.where(time.notna(), dte_time)


def _check_bounds(
    df: pd.DataFrame,
    work: pd.DataFrame,
    spot: float,
    tolerance: float,
    violations: list[dict[str, Any]],
    reason_buckets: dict[str, int],
) -> None:
    valid = work.dropna(subset=["price_num", "strike_num", "time_num"])
    for idx, row in valid.iterrows():
        option_type = row["type_norm"]
        if option_type not in {"call", "put"}:
            continue
        discounted_spot = float(spot * np.exp(-row["dividend_num"] * row["time_num"]))
        discounted_strike = float(row["strike_num"] * np.exp(-row["rate_num"] * row["time_num"]))
        if option_type == "call":
            lower = max(discounted_spot - discounted_strike, 0.0)
            upper = discounted_spot
        else:
            lower = max(discounted_strike - discounted_spot, 0.0)
            upper = discounted_strike
        df.loc[idx, "noArbitrageLowerBound"] = lower
        df.loc[idx, "noArbitrageUpperBound"] = upper
        price = float(row["price_num"])
        if price < lower - tolerance or price > upper + tolerance:
            _flag(
                df,
                [idx],
                "bounds",
                "noArbitrageBoundViolation",
                reason_buckets,
            )
            violations.append(
                {
                    "check": "bounds",
                    "expiration": _iso_date(row["expiration_norm"]),
                    "type": option_type,
                    "strike": float(row["strike_num"]),
                    "price": price,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "tolerance": tolerance,
                }
            )


def _check_strike_monotonicity(
    df: pd.DataFrame,
    work: pd.DataFrame,
    tolerance: float,
    violations: list[dict[str, Any]],
    reason_buckets: dict[str, int],
) -> None:
    required = ["expiration_norm", "type_norm", "strike_num", "price_num"]
    valid = work.dropna(subset=required)
    for (expiry, option_type), group in valid.groupby(["expiration_norm", "type_norm"], dropna=True):
        if option_type not in {"call", "put"}:
            continue
        group = group.sort_values("strike_num")
        for left, right in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            left_price = float(left.price_num)
            right_price = float(right.price_num)
            if option_type == "call" and right_price > left_price + tolerance:
                reason = "call_monotonicity"
            elif option_type == "put" and right_price < left_price - tolerance:
                reason = "put_monotonicity"
            else:
                continue
            _flag(df, [left.Index, right.Index], reason, "noArbitrageMonotonicityViolation", reason_buckets)
            violations.append(
                {
                    "check": reason,
                    "expiration": _iso_date(expiry),
                    "type": option_type,
                    "low_strike": float(left.strike_num),
                    "high_strike": float(right.strike_num),
                    "low_strike_price": left_price,
                    "high_strike_price": right_price,
                    "tolerance": tolerance,
                }
            )


def _check_convexity(
    df: pd.DataFrame,
    work: pd.DataFrame,
    tolerance: float,
    violations: list[dict[str, Any]],
    reason_buckets: dict[str, int],
) -> None:
    required = ["expiration_norm", "type_norm", "strike_num", "price_num"]
    valid = work.dropna(subset=required)
    for (expiry, option_type), group in valid.groupby(["expiration_norm", "type_norm"], dropna=True):
        if option_type not in {"call", "put"} or len(group) < 3:
            continue
        group = group.sort_values("strike_num")
        rows = list(group.itertuples())
        for left, middle, right in zip(rows[:-2], rows[1:-1], rows[2:]):
            span = float(right.strike_num - left.strike_num)
            if span <= 0:
                continue
            left_weight = float((right.strike_num - middle.strike_num) / span)
            right_weight = 1.0 - left_weight
            convex_upper = left_weight * float(left.price_num) + right_weight * float(right.price_num)
            middle_price = float(middle.price_num)
            if middle_price <= convex_upper + tolerance:
                continue
            _flag(df, [middle.Index], "convexity", "noArbitrageConvexityViolation", reason_buckets)
            violations.append(
                {
                    "check": "convexity",
                    "expiration": _iso_date(expiry),
                    "type": option_type,
                    "strike": float(middle.strike_num),
                    "price": middle_price,
                    "convex_upper": float(convex_upper),
                    "tolerance": tolerance,
                }
            )


def _check_calendar_total_variance(
    df: pd.DataFrame,
    work: pd.DataFrame,
    violations: list[dict[str, Any]],
    reason_buckets: dict[str, int],
) -> None:
    valid = work.dropna(subset=["type_norm", "strike_num", "time_num", "iv_num"])
    valid = valid[(valid["time_num"] > 0) & (valid["iv_num"] > 0)]
    for (option_type, strike), group in valid.groupby(["type_norm", "strike_num"], dropna=True):
        if option_type not in {"call", "put"} or len(group) < 2:
            continue
        group = group.sort_values("time_num")
        group = group.assign(total_variance=group["iv_num"] ** 2 * group["time_num"])
        for front, back in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            if float(back.total_variance) + 1e-8 >= float(front.total_variance):
                continue
            _flag(
                df,
                [front.Index, back.Index],
                "calendar_monotonicity",
                "noArbitrageCalendarViolation",
                reason_buckets,
            )
            violations.append(
                {
                    "check": "calendar_monotonicity",
                    "type": option_type,
                    "strike": float(strike),
                    "front_expiration": _iso_date(front.expiration_norm),
                    "back_expiration": _iso_date(back.expiration_norm),
                    "front_total_variance": float(front.total_variance),
                    "back_total_variance": float(back.total_variance),
                }
            )


def _flag(
    df: pd.DataFrame,
    indices: list[Any],
    reason: str,
    flag_column: str,
    reason_buckets: dict[str, int],
) -> None:
    reason_buckets[reason] = reason_buckets.get(reason, 0) + 1
    for idx in indices:
        df.loc[idx, "noArbitrageViolation"] = True
        df.loc[idx, flag_column] = True
        existing = str(df.loc[idx, "noArbitrageReasons"] or "")
        reasons = {item for item in existing.split("|") if item}
        reasons.add(reason)
        df.loc[idx, "noArbitrageReasons"] = "|".join(sorted(reasons))


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _iso_date(value: Any) -> str | None:
    converted = pd.to_datetime(value, errors="coerce")
    return converted.date().isoformat() if pd.notna(converted) else None
