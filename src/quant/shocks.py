"""Surface shock scenario analytics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.pricing.black_scholes import BlackScholesModel, OptionGreeks


DEFAULT_SURFACE_SHOCKS = (
    {"name": "Parallel +5 vol pts", "kind": "parallel_vol", "vol_shift": 0.05, "spot_shift": 0.0},
    {"name": "Parallel -5 vol pts", "kind": "parallel_vol", "vol_shift": -0.05, "spot_shift": 0.0},
    {"name": "Skew steepen", "kind": "skew", "vol_shift": 0.05, "spot_shift": 0.0},
    {"name": "Skew flatten", "kind": "skew", "vol_shift": -0.05, "spot_shift": 0.0},
    {"name": "Term twist front up", "kind": "term_twist", "vol_shift": 0.05, "spot_shift": 0.0},
    {"name": "Spot +5%", "kind": "spot", "vol_shift": 0.0, "spot_shift": 0.05},
    {"name": "Spot -5%", "kind": "spot", "vol_shift": 0.0, "spot_shift": -0.05},
)


def surface_shock_scenarios(
    frame: pd.DataFrame,
    spot: float,
    scenarios: tuple[dict[str, Any], ...] = DEFAULT_SURFACE_SHOCKS,
) -> dict[str, Any]:
    """Return unit-contract P&L and Greek impacts for deterministic surface shocks."""
    work = _prepared_frame(frame, spot)
    if work.empty:
        return {
            "available": False,
            "reason": "No option rows have usable price, IV, strike, and expiry inputs",
            "source": "current option chain",
            "position_assumption": "one long contract per option row",
            "scenarios": [],
        }

    current_prices = _price_vector(work, work["spot"], work["iv"])
    current_delta = _delta_vector(work, work["spot"], work["iv"])
    current_vega = _vega_vector(work, work["spot"], work["iv"])
    rows = []
    for scenario in scenarios:
        shocked_spot = work["spot"] * (1.0 + float(scenario.get("spot_shift", 0.0)))
        shocked_iv = _shocked_iv(work, scenario)
        shocked_prices = _price_vector(work, shocked_spot, shocked_iv)
        shocked_delta = _delta_vector(work, shocked_spot, shocked_iv)
        shocked_vega = _vega_vector(work, shocked_spot, shocked_iv)
        pnl = shocked_prices - current_prices
        rows.append(
            {
                "scenario": str(scenario["name"]),
                "spot_shift": float(scenario.get("spot_shift", 0.0)),
                "vol_shift": float(scenario.get("vol_shift", 0.0)),
                "contracts": int(len(work)),
                "unit_contract_pnl": float(np.nansum(pnl)),
                "mean_contract_pnl": float(np.nanmean(pnl)),
                "max_contract_loss": float(np.nanmin(pnl)),
                "max_contract_gain": float(np.nanmax(pnl)),
                "delta_before": float(np.nansum(current_delta)),
                "delta_after": float(np.nansum(shocked_delta)),
                "delta_change": float(np.nansum(shocked_delta - current_delta)),
                "vega_before": float(np.nansum(current_vega)),
                "vega_after": float(np.nansum(shocked_vega)),
                "vega_change": float(np.nansum(shocked_vega - current_vega)),
                "mean_shocked_iv": float(np.nanmean(shocked_iv)),
            }
        )

    return {
        "available": True,
        "source": "current option chain",
        "position_assumption": "one long contract per option row",
        "base_contracts": int(len(work)),
        "base_market_value": float(np.nansum(current_prices)),
        "base_delta": float(np.nansum(current_delta)),
        "base_vega": float(np.nansum(current_vega)),
        "scenarios": rows,
    }


def _prepared_frame(frame: pd.DataFrame, spot: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "type": frame.get("type", pd.Series("", index=frame.index)).astype(str).str.lower(),
            "strike": pd.to_numeric(frame.get("strike"), errors="coerce"),
            "dte": pd.to_numeric(frame.get("daysToExpiration"), errors="coerce"),
            "rate": pd.to_numeric(frame.get("riskFreeRate"), errors="coerce").fillna(0.0),
            "dividend": pd.to_numeric(
                frame.get("effectiveDividendYield", frame.get("dividendYield")),
                errors="coerce",
            ).fillna(0.0),
            "iv": pd.to_numeric(frame.get("computedIV", frame.get("impliedVolatility")), errors="coerce"),
            "log_moneyness": pd.to_numeric(frame.get("logMoneyness"), errors="coerce"),
        }
    )
    out["spot"] = float(spot)
    out["time"] = out["dte"] / 365.0
    fallback_log_money = np.log(out["strike"] / float(spot)) if spot > 0 else np.nan
    out["log_moneyness"] = out["log_moneyness"].where(out["log_moneyness"].notna(), fallback_log_money)
    out = out.dropna(subset=["strike", "time", "iv"])
    out = out[(out["strike"] > 0.0) & (out["time"] > 0.0) & (out["iv"] > 0.0)]
    out = out[out["type"].isin({"call", "put"})]
    return out.reset_index(drop=True)


def _shocked_iv(work: pd.DataFrame, scenario: dict[str, Any]) -> pd.Series:
    kind = str(scenario.get("kind"))
    shift = float(scenario.get("vol_shift", 0.0))
    if kind == "skew":
        log_money = work["log_moneyness"].clip(-0.30, 0.30) / 0.30
        adjustment = -shift * log_money
    elif kind == "term_twist":
        dte = work["dte"]
        span = max(float(dte.max() - dte.min()), 1.0)
        term_score = 1.0 - 2.0 * (dte - float(dte.min())) / span
        adjustment = shift * term_score
    else:
        adjustment = shift
    return (work["iv"] + adjustment).clip(lower=0.01, upper=5.0)


def _price_vector(work: pd.DataFrame, spot: pd.Series, iv: pd.Series) -> np.ndarray:
    return np.array(
        [
            BlackScholesModel.option_price(s, k, t, r, v, option, q)
            for s, k, t, r, v, option, q in zip(
                spot,
                work["strike"],
                work["time"],
                work["rate"],
                iv,
                work["type"],
                work["dividend"],
            )
        ],
        dtype=float,
    )


def _delta_vector(work: pd.DataFrame, spot: pd.Series, iv: pd.Series) -> np.ndarray:
    return np.array(
        [
            OptionGreeks.delta(s, k, t, r, v, option, q)
            for s, k, t, r, v, option, q in zip(
                spot,
                work["strike"],
                work["time"],
                work["rate"],
                iv,
                work["type"],
                work["dividend"],
            )
        ],
        dtype=float,
    )


def _vega_vector(work: pd.DataFrame, spot: pd.Series, iv: pd.Series) -> np.ndarray:
    return np.array(
        [
            OptionGreeks.vega(s, k, t, r, v, q)
            for s, k, t, r, v, q in zip(
                spot,
                work["strike"],
                work["time"],
                work["rate"],
                iv,
                work["dividend"],
            )
        ],
        dtype=float,
    )
