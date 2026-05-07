from datetime import datetime

import pandas as pd
import pytest

from src.data.models import MarketDataSnapshot
from src.pricing.black_scholes import BlackScholesModel
from src.quant.heston import calibrate_heston_from_snapshot, calibrate_heston_research
from src.quant.model_selection import apply_model_selection, contract_greeks_metadata
from src.quant.sabr import calibrate_sabr_by_expiry


def _research_chain() -> pd.DataFrame:
    rows = []
    spot = 100.0
    for expiration, dte, base_iv in (
        (pd.Timestamp("2026-06-19"), 45, 0.22),
        (pd.Timestamp("2026-09-18"), 136, 0.25),
    ):
        for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
            log_money = pd.NA if strike == 100.0 else __import__("math").log(strike / spot)
            skew = -0.04 * ((strike / spot) - 1.0)
            iv = base_iv + skew + 0.02 * ((strike / spot) - 1.0) ** 2
            rows.append(
                {
                    "type": "call",
                    "expiration": expiration,
                    "daysToExpiration": dte,
                    "strike": strike,
                    "selectedMarketPrice": BlackScholesModel.call_price(
                        spot,
                        strike,
                        dte / 365.0,
                        0.03,
                        iv,
                        0.01,
                    ),
                    "computedIV": iv,
                    "impliedVolatility": iv,
                    "riskFreeRate": 0.03,
                    "effectiveDividendYield": 0.01,
                    "forwardPrice": spot,
                    "logMoneyness": log_money,
                }
            )
    return pd.DataFrame(rows)


def test_model_selection_adds_visible_model_prices_and_contract_greeks():
    chain = _research_chain().head(3)
    priced = apply_model_selection(chain, 100.0, "BSM with dividends")

    assert {"pricingModel", "selectedModelPrice", "selectedModelResidual", "delta", "gamma", "theta", "vega", "rho"}.issubset(
        priced.columns
    )
    assert priced["pricingModel"].unique().tolist() == ["BSM with dividends"]
    assert priced["selectedModelPrice"].notna().all()
    assert priced["delta"].notna().all()
    assert priced["vega"].notna().all()

    meta = contract_greeks_metadata(priced, "BSM with dividends")
    assert meta["pricing_model_label"] == "BSM with dividends"
    assert meta["contract_greeks_count"] == 3
    assert meta["greek_units"]["vega"] == "option dollars per one volatility-point move"


def test_binomial_model_selection_uses_american_pricing_path():
    chain = _research_chain().head(1).copy()
    chain.loc[chain.index[0], "type"] = "put"
    chain.loc[chain.index[0], "strike"] = 110.0
    chain.loc[chain.index[0], "selectedMarketPrice"] = 11.0

    priced = apply_model_selection(chain, 100.0, "CRR binomial", steps=150)

    assert priced.loc[0, "pricingModel"] == "CRR binomial"
    assert priced.loc[0, "selectedModelPrice"] >= BlackScholesModel.put_price(
        100.0,
        110.0,
        45 / 365.0,
        0.03,
        priced.loc[0, "computedIV"],
        0.01,
    )


def test_heston_research_calibration_reports_fit_errors_and_warnings():
    result = calibrate_heston_research(_research_chain(), 100.0)

    assert result["status"] == "fitted"
    assert result["points"] == 10
    assert result["rmse"] is not None
    assert result["warnings"]
    assert result["parameterization"] == "variance_dynamics_surrogate"


def test_heston_research_can_run_from_snapshot_object():
    snapshot = MarketDataSnapshot.from_chain_frame(
        "SPY",
        100.0,
        datetime(2026, 5, 7, 12, 0, 0),
        _research_chain(),
        {"source": "unit fixture", "mode": "Synthetic", "valid_rows": 10, "pricing_model": "bsm_dividends"},
    )

    result = calibrate_heston_from_snapshot(snapshot)

    assert result["status"] == "fitted"
    assert result["snapshot_symbol"] == "SPY"
    assert result["snapshot_source"] == "unit fixture"


def test_sabr_is_optional_for_non_index_symbols_and_fits_index_smiles():
    skipped = calibrate_sabr_by_expiry(_research_chain(), 100.0, symbol="AAPL")
    fitted = calibrate_sabr_by_expiry(_research_chain(), 100.0, symbol="SPY")

    assert skipped["status"] == "skipped"
    assert fitted["status"] == "fitted"
    assert fitted["fitted_expiries"] == 2
    assert fitted["rmse"] == pytest.approx(fitted["rmse"])
