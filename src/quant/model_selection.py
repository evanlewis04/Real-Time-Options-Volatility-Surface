"""Pricing model selection and contract-level analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.pricing.black_scholes import BlackScholesModel, OptionGreeks
from src.quant.american import binomial_american_price


BSM = "bsm"
BSM_DIVIDENDS = "bsm_dividends"
BINOMIAL = "binomial"
HESTON_RESEARCH = "heston_research"

MODEL_LABELS = {
    BSM: "BSM",
    BSM_DIVIDENDS: "BSM with dividends",
    BINOMIAL: "CRR binomial",
    HESTON_RESEARCH: "Heston research placeholder",
}

MODEL_CHOICES = tuple(MODEL_LABELS.values())


@dataclass(frozen=True)
class ContractInputs:
    spot: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    option_type: str
    dividend_yield: float


def normalize_pricing_model(value: str | None) -> str:
    """Normalize dashboard model labels and stored model keys."""
    text = str(value or BSM_DIVIDENDS).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bsm": BSM,
        "black_scholes": BSM,
        "black_scholes_merton": BSM_DIVIDENDS,
        "bsm_with_dividends": BSM_DIVIDENDS,
        "dividend_bsm": BSM_DIVIDENDS,
        "crr_binomial": BINOMIAL,
        "binomial": BINOMIAL,
        "heston": HESTON_RESEARCH,
        "heston_research": HESTON_RESEARCH,
        "heston_research_placeholder": HESTON_RESEARCH,
    }
    label_aliases = {label.lower().replace(" ", "_"): key for key, label in MODEL_LABELS.items()}
    aliases.update(label_aliases)
    return aliases.get(text, BSM_DIVIDENDS)


def pricing_model_metadata(model: str | None) -> dict[str, Any]:
    """Return model provenance shown in dashboard metadata and persisted snapshots."""
    key = normalize_pricing_model(model)
    assumptions = {
        BSM: "European Black-Scholes with continuous compounding and zero dividend yield.",
        BSM_DIVIDENDS: "European Black-Scholes-Merton with expiry-specific effective dividend yield.",
        BINOMIAL: "American Cox-Ross-Rubinstein tree with expiry-specific rates and dividends.",
        HESTON_RESEARCH: (
            "Research-only Heston calibration is reported separately; contract pricing uses "
            "BSM with dividends until production Heston pricing is implemented."
        ),
    }
    warnings = {
        HESTON_RESEARCH: "Heston is calibration-only in this release; contract prices fall back to BSM with dividends.",
    }
    return {
        "pricing_model": key,
        "pricing_model_label": MODEL_LABELS[key],
        "pricing_model_assumptions": assumptions[key],
        "pricing_model_warning": warnings.get(key),
        "available_pricing_models": list(MODEL_LABELS.values()),
    }


def price_contract(inputs: ContractInputs, model: str | None, *, steps: int = 100) -> float:
    """Price one option under the selected dashboard model."""
    key = normalize_pricing_model(model)
    if not _valid_contract_inputs(inputs):
        return np.nan

    dividend = inputs.dividend_yield if key in {BSM_DIVIDENDS, BINOMIAL, HESTON_RESEARCH} else 0.0
    if key == BINOMIAL:
        return binomial_american_price(
            inputs.spot,
            inputs.strike,
            inputs.time_to_expiry,
            inputs.risk_free_rate,
            inputs.volatility,
            inputs.option_type,
            dividend_yield=dividend,
            steps=steps,
        )

    return float(
        BlackScholesModel.option_price(
            inputs.spot,
            inputs.strike,
            inputs.time_to_expiry,
            inputs.risk_free_rate,
            inputs.volatility,
            inputs.option_type,
            dividend,
        )
    )


def apply_model_selection(frame: pd.DataFrame, spot: float, model: str | None, *, steps: int = 100) -> pd.DataFrame:
    """Attach selected-model prices, residuals, and contract-level Greeks."""
    if frame.empty:
        return frame.copy()

    key = normalize_pricing_model(model)
    out = frame.copy()
    prices: list[float] = []
    deltas: list[float] = []
    gammas: list[float] = []
    thetas: list[float] = []
    vegas: list[float] = []
    rhos: list[float] = []
    vannas: list[float] = []
    volgas: list[float] = []
    vommas: list[float] = []
    charms: list[float] = []
    speeds: list[float] = []
    colors: list[float] = []

    for _, row in out.iterrows():
        inputs = _contract_inputs(row, spot)
        price = price_contract(inputs, key, steps=steps)
        greek = _contract_greeks(inputs, key, steps=steps)
        prices.append(price)
        deltas.append(greek["delta"])
        gammas.append(greek["gamma"])
        thetas.append(greek["theta"])
        vegas.append(greek["vega"])
        rhos.append(greek["rho"])
        vannas.append(greek["vanna"])
        volgas.append(greek["volga"])
        vommas.append(greek["vomma"])
        charms.append(greek["charm"])
        speeds.append(greek["speed"])
        colors.append(greek["color"])

    out["pricingModel"] = MODEL_LABELS[key]
    out["selectedModelPrice"] = prices
    market_price = pd.to_numeric(
        out.get("selectedMarketPrice", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out["selectedModelResidual"] = market_price - out["selectedModelPrice"]
    out["delta"] = deltas
    out["gamma"] = gammas
    out["theta"] = thetas
    out["vega"] = vegas
    out["rho"] = rhos
    out["vanna"] = vannas
    out["volga"] = volgas
    out["vomma"] = vommas
    out["charm"] = charms
    out["speed"] = speeds
    out["color"] = colors
    if key == HESTON_RESEARCH:
        out["pricingModelWarning"] = pricing_model_metadata(key)["pricing_model_warning"]
    return out


def contract_greeks_metadata(frame: pd.DataFrame, model: str | None) -> dict[str, Any]:
    """Summarize contract-level Greek availability and units."""
    if frame.empty or "delta" not in frame:
        return {
            **pricing_model_metadata(model),
            "contract_greeks_count": 0,
            "second_order_greeks_count": 0,
            "greek_units": _greek_units(),
        }
    count = int(pd.to_numeric(frame.get("delta"), errors="coerce").notna().sum())
    second_order_count = int(pd.to_numeric(frame.get("vanna"), errors="coerce").notna().sum())
    residuals = pd.to_numeric(frame.get("selectedModelResidual"), errors="coerce").dropna()
    return {
        **pricing_model_metadata(model),
        "contract_greeks_count": count,
        "second_order_greeks_count": second_order_count,
        "greek_units": _greek_units(),
        "median_selected_model_residual": float(residuals.median()) if not residuals.empty else None,
        "max_abs_selected_model_residual": float(residuals.abs().max()) if not residuals.empty else None,
    }


def _contract_greeks(inputs: ContractInputs, model: str, *, steps: int) -> dict[str, float]:
    if not _valid_contract_inputs(inputs):
        return _empty_greeks()

    if model in {BSM, BSM_DIVIDENDS, HESTON_RESEARCH}:
        dividend = 0.0 if model == BSM else inputs.dividend_yield
        return {
            "delta": float(
                OptionGreeks.delta(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    inputs.option_type,
                    dividend,
                )
            ),
            "gamma": float(
                OptionGreeks.gamma(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    dividend,
                )
            ),
            "theta": float(
                OptionGreeks.theta(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    inputs.option_type,
                    dividend,
                )
            ),
            "vega": float(
                OptionGreeks.vega(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    dividend,
                )
            ),
            "rho": float(
                OptionGreeks.rho(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    inputs.option_type,
                    dividend,
                )
            ),
            "vanna": float(
                OptionGreeks.vanna(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    dividend,
                )
            ),
            "volga": float(
                OptionGreeks.volga(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    dividend,
                )
            ),
            "vomma": float(
                OptionGreeks.vomma(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    dividend,
                )
            ),
            "charm": float(
                OptionGreeks.charm(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    inputs.option_type,
                    dividend,
                )
            ),
            "speed": float(
                OptionGreeks.speed(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    dividend,
                )
            ),
            "color": float(
                OptionGreeks.color(
                    inputs.spot,
                    inputs.strike,
                    inputs.time_to_expiry,
                    inputs.risk_free_rate,
                    inputs.volatility,
                    dividend,
                )
            ),
        }

    return _finite_difference_greeks(inputs, model, steps=steps)


def _finite_difference_greeks(inputs: ContractInputs, model: str, *, steps: int) -> dict[str, float]:
    spot_step = max(inputs.spot * 0.01, 0.01)
    vol_step = 0.01
    rate_step = 0.01
    day = 1.0 / 365.0

    base = price_contract(inputs, model, steps=steps)
    up_spot = price_contract(_replace_inputs(inputs, spot=inputs.spot + spot_step), model, steps=steps)
    down_spot = price_contract(_replace_inputs(inputs, spot=max(inputs.spot - spot_step, 0.01)), model, steps=steps)
    up_vol = price_contract(_replace_inputs(inputs, volatility=inputs.volatility + vol_step), model, steps=steps)
    down_vol = price_contract(
        _replace_inputs(inputs, volatility=max(inputs.volatility - vol_step, 0.0001)),
        model,
        steps=steps,
    )
    up_rate = price_contract(_replace_inputs(inputs, risk_free_rate=inputs.risk_free_rate + rate_step), model, steps=steps)
    shorter = price_contract(
        _replace_inputs(inputs, time_to_expiry=max(inputs.time_to_expiry - day, 0.0)),
        model,
        steps=steps,
    )
    delta_now = (up_spot - down_spot) / (2.0 * spot_step)
    gamma_now = (up_spot - 2.0 * base + down_spot) / (spot_step**2)
    shorter_inputs = _replace_inputs(inputs, time_to_expiry=max(inputs.time_to_expiry - day, 0.0))
    shorter_up_spot = price_contract(_replace_inputs(shorter_inputs, spot=inputs.spot + spot_step), model, steps=steps)
    shorter_down_spot = price_contract(
        _replace_inputs(shorter_inputs, spot=max(inputs.spot - spot_step, 0.01)),
        model,
        steps=steps,
    )
    delta_shorter = (shorter_up_spot - shorter_down_spot) / (2.0 * spot_step)
    gamma_shorter = (shorter_up_spot - 2.0 * shorter + shorter_down_spot) / (spot_step**2)
    up_spot_up_vol = price_contract(
        _replace_inputs(inputs, spot=inputs.spot + spot_step, volatility=inputs.volatility + vol_step),
        model,
        steps=steps,
    )
    down_spot_up_vol = price_contract(
        _replace_inputs(inputs, spot=max(inputs.spot - spot_step, 0.01), volatility=inputs.volatility + vol_step),
        model,
        steps=steps,
    )
    up_spot_down_vol = price_contract(
        _replace_inputs(
            inputs,
            spot=inputs.spot + spot_step,
            volatility=max(inputs.volatility - vol_step, 0.0001),
        ),
        model,
        steps=steps,
    )
    down_spot_down_vol = price_contract(
        _replace_inputs(
            inputs,
            spot=max(inputs.spot - spot_step, 0.01),
            volatility=max(inputs.volatility - vol_step, 0.0001),
        ),
        model,
        steps=steps,
    )
    raw_vanna = ((up_spot_up_vol - down_spot_up_vol) - (up_spot_down_vol - down_spot_down_vol)) / (
        4.0 * spot_step * vol_step
    )
    raw_volga = (up_vol - 2.0 * base + down_vol) / (vol_step**2)
    return {
        "delta": delta_now,
        "gamma": gamma_now,
        "theta": shorter - base,
        "vega": up_vol - base,
        "rho": up_rate - base,
        "vanna": raw_vanna / 100.0,
        "volga": raw_volga / 10000.0,
        "vomma": raw_volga / 10000.0,
        "charm": delta_shorter - delta_now,
        "speed": (
            price_contract(_replace_inputs(inputs, spot=inputs.spot + 2.0 * spot_step), model, steps=steps)
            - 2.0 * up_spot
            + 2.0 * down_spot
            - price_contract(
                _replace_inputs(inputs, spot=max(inputs.spot - 2.0 * spot_step, 0.01)),
                model,
                steps=steps,
            )
        )
        / (2.0 * spot_step**3),
        "color": gamma_shorter - gamma_now,
    }


def _empty_greeks() -> dict[str, float]:
    return {
        "delta": np.nan,
        "gamma": np.nan,
        "theta": np.nan,
        "vega": np.nan,
        "rho": np.nan,
        "vanna": np.nan,
        "volga": np.nan,
        "vomma": np.nan,
        "charm": np.nan,
        "speed": np.nan,
        "color": np.nan,
    }


def _contract_inputs(row: pd.Series, spot: float) -> ContractInputs:
    dte = _num(row.get("daysToExpiration"))
    time_to_expiry = _num(row.get("time_to_expiry"))
    if not np.isfinite(time_to_expiry):
        time_to_expiry = dte / 365.0 if np.isfinite(dte) else np.nan
    dividend = _num(row.get("effectiveDividendYield"), default=_num(row.get("dividendYield"), default=0.0))
    return ContractInputs(
        spot=float(spot),
        strike=_num(row.get("strike")),
        time_to_expiry=time_to_expiry,
        risk_free_rate=_num(row.get("riskFreeRate"), default=0.0),
        volatility=_num(row.get("computedIV"), default=_num(row.get("impliedVolatility"))),
        option_type=str(row.get("type", "")).lower(),
        dividend_yield=dividend if np.isfinite(dividend) else 0.0,
    )


def _replace_inputs(inputs: ContractInputs, **changes: float) -> ContractInputs:
    data = {
        "spot": inputs.spot,
        "strike": inputs.strike,
        "time_to_expiry": inputs.time_to_expiry,
        "risk_free_rate": inputs.risk_free_rate,
        "volatility": inputs.volatility,
        "option_type": inputs.option_type,
        "dividend_yield": inputs.dividend_yield,
    }
    data.update(changes)
    return ContractInputs(**data)


def _valid_contract_inputs(inputs: ContractInputs) -> bool:
    return (
        inputs.option_type in {"call", "put"}
        and all(
            np.isfinite(value)
            for value in (
                inputs.spot,
                inputs.strike,
                inputs.time_to_expiry,
                inputs.risk_free_rate,
                inputs.volatility,
                inputs.dividend_yield,
            )
        )
        and inputs.spot > 0.0
        and inputs.strike > 0.0
        and inputs.time_to_expiry >= 0.0
        and inputs.volatility >= 0.0
    )


def _greek_units() -> dict[str, str]:
    return {
        "delta": "option dollars per $1 spot move",
        "gamma": "delta change per $1 spot move",
        "theta": "option dollars per calendar day",
        "vega": "option dollars per one volatility-point move",
        "rho": "option dollars per one percentage-point rate move",
        "vanna": "delta change per one volatility-point move",
        "volga": "Vega/1% change per one volatility-point move",
        "vomma": "Vega/1% change per one volatility-point move",
        "charm": "delta change per calendar day",
        "speed": "gamma change per $1 spot move",
        "color": "gamma change per calendar day",
    }


def _num(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
