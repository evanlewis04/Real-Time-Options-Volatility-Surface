"""Option price decomposition helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.pricing.black_scholes import BlackScholesModel


def apply_price_decomposition(frame: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Attach intrinsic, time, carry, vol contribution, and model residual fields."""
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    rows: dict[str, list[float]] = {
        "intrinsicValue": [],
        "timeValue": [],
        "carryValue": [],
        "impliedVolContribution": [],
        "modelResidual": [],
        "zeroVolPrice": [],
        "bsmPrice": [],
    }

    for _, row in out.iterrows():
        option_type = str(row.get("type", "")).lower()
        strike = _num(row.get("strike"))
        market = _market_price(row)
        dte = _num(row.get("daysToExpiration"))
        time_to_expiry = _num(row.get("time_to_expiry"))
        if not np.isfinite(time_to_expiry):
            time_to_expiry = dte / 365.0 if np.isfinite(dte) else np.nan
        rate = _num(row.get("riskFreeRate"), default=0.0)
        dividend = _num(row.get("effectiveDividendYield"), default=_num(row.get("dividendYield"), default=0.0))
        iv = _num(row.get("computedIV"), default=_num(row.get("impliedVolatility")))

        intrinsic = _intrinsic(spot, strike, option_type)
        zero_vol = _zero_vol_price(spot, strike, time_to_expiry, rate, dividend, option_type)
        bsm = _bsm_price(spot, strike, time_to_expiry, rate, iv, dividend, option_type)
        time_value = market - intrinsic if np.isfinite(market) and np.isfinite(intrinsic) else np.nan
        carry = zero_vol - intrinsic if np.isfinite(zero_vol) and np.isfinite(intrinsic) else np.nan
        vol_contribution = bsm - zero_vol if np.isfinite(bsm) and np.isfinite(zero_vol) else np.nan
        residual = market - bsm if np.isfinite(market) and np.isfinite(bsm) else np.nan

        rows["intrinsicValue"].append(intrinsic)
        rows["timeValue"].append(time_value)
        rows["carryValue"].append(carry)
        rows["impliedVolContribution"].append(vol_contribution)
        rows["modelResidual"].append(residual)
        rows["zeroVolPrice"].append(zero_vol)
        rows["bsmPrice"].append(bsm)

    for column, values in rows.items():
        out[column] = values
    out["decompositionPrice"] = out.get("selectedMarketPrice")
    return out


def price_decomposition_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize decomposition availability and residuals."""
    if frame.empty or "intrinsicValue" not in frame:
        return {
            "price_decomposition_contracts": 0,
            "price_decomposition_source": "selected market price",
            "median_time_value": None,
            "median_model_residual": None,
        }
    residuals = pd.to_numeric(frame.get("modelResidual"), errors="coerce").dropna()
    time_values = pd.to_numeric(frame.get("timeValue"), errors="coerce").dropna()
    return {
        "price_decomposition_contracts": int(pd.to_numeric(frame.get("intrinsicValue"), errors="coerce").notna().sum()),
        "price_decomposition_source": "selected market price",
        "median_time_value": float(time_values.median()) if not time_values.empty else None,
        "median_model_residual": float(residuals.median()) if not residuals.empty else None,
        "max_abs_model_residual": float(residuals.abs().max()) if not residuals.empty else None,
    }


def _market_price(row: pd.Series) -> float:
    for column in ("selectedMarketPrice", "mark", "mid", "last"):
        value = _num(row.get(column))
        if np.isfinite(value) and value > 0:
            return value
    return np.nan


def _intrinsic(spot: float, strike: float, option_type: str) -> float:
    if spot <= 0 or not np.isfinite(strike):
        return np.nan
    if option_type == "call":
        return float(max(spot - strike, 0.0))
    if option_type == "put":
        return float(max(strike - spot, 0.0))
    return np.nan


def _zero_vol_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    dividend: float,
    option_type: str,
) -> float:
    if not all(np.isfinite(value) for value in (spot, strike, time_to_expiry, rate, dividend)):
        return np.nan
    if option_type not in {"call", "put"}:
        return np.nan
    return float(BlackScholesModel.option_price(spot, strike, time_to_expiry, rate, 0.0, option_type, dividend))


def _bsm_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    iv: float,
    dividend: float,
    option_type: str,
) -> float:
    if not all(np.isfinite(value) for value in (spot, strike, time_to_expiry, rate, iv, dividend)):
        return np.nan
    if option_type not in {"call", "put"}:
        return np.nan
    return float(BlackScholesModel.option_price(spot, strike, time_to_expiry, rate, iv, option_type, dividend))


def _num(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
