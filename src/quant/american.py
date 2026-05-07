"""American option pricing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.pricing.black_scholes import BlackScholesModel


def binomial_american_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str,
    dividend_yield: float = 0.0,
    steps: int = 100,
) -> float:
    """Price an American option with a Cox-Ross-Rubinstein tree."""
    option = str(option_type).lower()
    if option not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if spot <= 0 or strike <= 0:
        return np.nan
    if time_to_expiry <= 0:
        return _payoff(spot, strike, option)
    if volatility <= 0:
        return _payoff(spot, strike, option)

    n_steps = max(1, int(steps))
    dt = float(time_to_expiry) / n_steps
    up = float(np.exp(volatility * np.sqrt(dt)))
    down = 1.0 / up
    growth = float(np.exp((risk_free_rate - dividend_yield) * dt))
    probability = (growth - down) / (up - down)
    probability = float(np.clip(probability, 0.0, 1.0))
    discount = float(np.exp(-risk_free_rate * dt))

    node_index = np.arange(n_steps + 1)
    terminal_spots = spot * (up ** (n_steps - node_index)) * (down ** node_index)
    values = _payoff_array(terminal_spots, strike, option)

    for step in range(n_steps - 1, -1, -1):
        values = discount * (probability * values[:-1] + (1.0 - probability) * values[1:])
        step_index = np.arange(step + 1)
        step_spots = spot * (up ** (step - step_index)) * (down ** step_index)
        values = np.maximum(values, _payoff_array(step_spots, strike, option))

    return float(max(values[0], 0.0))


def apply_american_pricing(frame: pd.DataFrame, spot: float, *, steps: int = 100) -> pd.DataFrame:
    """Attach European, American, and early-exercise premium columns."""
    if frame.empty:
        return frame.copy()

    out = frame.copy()
    american_prices = []
    european_prices = []
    premiums = []
    flags = []

    for _, row in out.iterrows():
        strike = _num(row.get("strike"))
        dte = _num(row.get("daysToExpiration"))
        time_to_expiry = _num(row.get("time_to_expiry"))
        if not np.isfinite(time_to_expiry):
            time_to_expiry = dte / 365.0 if np.isfinite(dte) else np.nan
        rate = _num(row.get("riskFreeRate"), default=0.0)
        dividend = _num(row.get("effectiveDividendYield"), default=_num(row.get("dividendYield"), default=0.0))
        iv = _num(row.get("computedIV"), default=_num(row.get("impliedVolatility")))
        option_type = str(row.get("type", "")).lower()

        if not all(np.isfinite(value) for value in (strike, time_to_expiry, rate, dividend, iv)):
            american_prices.append(np.nan)
            european_prices.append(np.nan)
            premiums.append(np.nan)
            flags.append(False)
            continue

        european = BlackScholesModel.option_price(spot, strike, time_to_expiry, rate, iv, option_type, dividend)
        american = binomial_american_price(
            spot,
            strike,
            time_to_expiry,
            rate,
            iv,
            option_type,
            dividend_yield=dividend,
            steps=steps,
        )
        premium = max(0.0, american - european) if np.isfinite(american) else np.nan
        european_prices.append(float(european))
        american_prices.append(float(american))
        premiums.append(float(premium) if np.isfinite(premium) else np.nan)
        flags.append(bool(np.isfinite(premium) and premium > 0.01))

    out["europeanPrice"] = european_prices
    out["americanPrice"] = american_prices
    out["earlyExercisePremium"] = premiums
    out["earlyExerciseFlag"] = flags
    out["americanModel"] = f"CRR binomial ({int(steps)} steps)"
    out["americanSteps"] = int(steps)
    return out


def american_pricing_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize American-vs-European pricing diagnostics."""
    if frame.empty or "americanPrice" not in frame:
        return {
            "american_model": "CRR binomial",
            "american_contracts_priced": 0,
            "early_exercise_candidates": 0,
            "median_early_exercise_premium": None,
            "max_early_exercise_premium": None,
        }
    premiums = pd.to_numeric(frame.get("earlyExercisePremium"), errors="coerce").dropna()
    return {
        "american_model": str(frame.get("americanModel", pd.Series(["CRR binomial"])).dropna().iloc[0]),
        "american_contracts_priced": int(pd.to_numeric(frame.get("americanPrice"), errors="coerce").notna().sum()),
        "early_exercise_candidates": int(pd.Series(frame.get("earlyExerciseFlag", False)).fillna(False).astype(bool).sum()),
        "median_early_exercise_premium": float(premiums.median()) if not premiums.empty else None,
        "max_early_exercise_premium": float(premiums.max()) if not premiums.empty else None,
    }


def _payoff(spot: float, strike: float, option_type: str) -> float:
    return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)


def _payoff_array(spots: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(spots - strike, 0.0)
    return np.maximum(strike - spots, 0.0)


def _num(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
