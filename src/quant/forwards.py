"""Forward-price and moneyness helpers for option analytics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def year_fraction_from_dte(dte: float | int | None) -> float:
    """Convert calendar days to a non-negative ACT/365 year fraction."""
    if dte is None or pd.isna(dte):
        return 0.0
    return max(0.0, float(dte) / 365.0)


def discount_factor(rate: float | int | None, dte: float | int | None) -> float:
    """Continuously compounded discount factor for a maturity."""
    rate_value = 0.0 if rate is None or pd.isna(rate) else float(rate)
    return float(np.exp(-rate_value * year_fraction_from_dte(dte)))


def forward_price(
    spot: float,
    dte: float | int | None,
    risk_free_rate: float | int | None,
    dividend_yield: float | int | None = 0.0,
    discrete_dividend_pv: float | int | None = 0.0,
) -> float:
    """Return the forward price implied by spot, rates, dividends, and maturity."""
    if spot <= 0:
        return np.nan
    rate_value = 0.0 if risk_free_rate is None or pd.isna(risk_free_rate) else float(risk_free_rate)
    dividend_value = 0.0 if dividend_yield is None or pd.isna(dividend_yield) else float(dividend_yield)
    dividend_pv = 0.0 if discrete_dividend_pv is None or pd.isna(discrete_dividend_pv) else float(discrete_dividend_pv)
    adjusted_spot = max(float(spot) - max(0.0, dividend_pv), 1e-9)
    return float(adjusted_spot * np.exp((rate_value - dividend_value) * year_fraction_from_dte(dte)))


def apply_forward_metrics(frame: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Attach discount, forward, forward-moneyness, and log-moneyness columns."""
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    dte = _numeric_column(enriched, "daysToExpiration")
    strikes = _numeric_column(enriched, "strike")
    rates = _numeric_column(enriched, "riskFreeRate")
    dividends = _numeric_column(enriched, "effectiveDividendYield")
    if dividends.isna().all() and "dividendYield" in enriched:
        dividends = _numeric_column(enriched, "dividendYield")

    forwards = [
        forward_price(spot, days, rate, dividend, 0.0)
        for days, rate, dividend in zip(dte, rates, dividends)
    ]
    discounts = [discount_factor(rate, days) for days, rate in zip(dte, rates)]

    enriched["discountFactor"] = discounts
    enriched["forwardPrice"] = forwards
    enriched["spotMoneyness"] = strikes / float(spot) if spot > 0 else np.nan
    enriched["forwardMoneyness"] = strikes / pd.Series(forwards, index=enriched.index).replace(0, np.nan)
    enriched["logMoneyness"] = np.log(enriched["forwardMoneyness"].where(enriched["forwardMoneyness"] > 0))
    return enriched


def expiry_forward_metadata(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build expiry-level forward and discount metadata for diagnostics."""
    if frame.empty or "expiration" not in frame.columns:
        return {}

    expiries = pd.to_datetime(frame["expiration"], errors="coerce")
    out: dict[str, dict[str, float]] = {}
    for expiry in sorted(expiries.dropna().dt.date.unique()):
        sub = frame[expiries.dt.date == expiry]
        forwards = pd.to_numeric(sub.get("forwardPrice"), errors="coerce").dropna()
        discounts = pd.to_numeric(sub.get("discountFactor"), errors="coerce").dropna()
        if forwards.empty and discounts.empty:
            continue
        out[expiry.isoformat()] = {
            "forward_price": float(forwards.median()) if not forwards.empty else np.nan,
            "discount_factor": float(discounts.median()) if not discounts.empty else np.nan,
        }
    return out


def clean_float(value: Any) -> float:
    """Convert optional tabular values to float or NaN."""
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")
