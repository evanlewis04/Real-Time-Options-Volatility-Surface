import pandas as pd
import pytest

from dashboard_connector import DashboardConnector
from src.pricing.black_scholes import BlackScholesModel
from src.quant.american import apply_american_pricing, binomial_american_price
from src.quant.price_decomposition import apply_price_decomposition
from src.quant.shocks import surface_shock_scenarios


def _chain() -> pd.DataFrame:
    call_price = BlackScholesModel.call_price(100.0, 100.0, 30 / 365, 0.03, 0.20, 0.01)
    put_price = BlackScholesModel.put_price(100.0, 100.0, 30 / 365, 0.03, 0.22, 0.01)
    return pd.DataFrame(
        [
            {
                "type": "call",
                "expiration": pd.Timestamp("2026-06-19"),
                "daysToExpiration": 30,
                "strike": 100.0,
                "selectedMarketPrice": call_price,
                "computedIV": 0.20,
                "riskFreeRate": 0.03,
                "effectiveDividendYield": 0.01,
                "logMoneyness": 0.0,
            },
            {
                "type": "put",
                "expiration": pd.Timestamp("2026-06-19"),
                "daysToExpiration": 30,
                "strike": 100.0,
                "selectedMarketPrice": put_price,
                "computedIV": 0.22,
                "riskFreeRate": 0.03,
                "effectiveDividendYield": 0.01,
                "logMoneyness": 0.0,
            },
            {
                "type": "call",
                "expiration": pd.Timestamp("2026-09-18"),
                "daysToExpiration": 121,
                "strike": 110.0,
                "selectedMarketPrice": 2.50,
                "computedIV": 0.25,
                "riskFreeRate": 0.03,
                "effectiveDividendYield": 0.01,
                "logMoneyness": 0.09531,
            },
        ]
    )


def test_surface_shock_scenarios_report_unit_contract_pnl_and_greeks():
    shocks = surface_shock_scenarios(_chain(), 100.0)

    assert shocks["available"]
    assert shocks["position_assumption"] == "one long contract per option row"
    assert shocks["base_contracts"] == 3
    parallel_up = next(row for row in shocks["scenarios"] if row["scenario"] == "Parallel +5 vol pts")
    parallel_down = next(row for row in shocks["scenarios"] if row["scenario"] == "Parallel -5 vol pts")
    assert parallel_up["unit_contract_pnl"] > 0.0
    assert parallel_down["unit_contract_pnl"] < 0.0
    assert "delta_change" in parallel_up
    assert "vega_change" in parallel_up


def test_price_decomposition_explains_selected_market_price():
    decomposed = apply_price_decomposition(_chain().head(1), 100.0)
    row = decomposed.iloc[0]

    assert row["intrinsicValue"] == pytest.approx(0.0)
    assert row["timeValue"] == pytest.approx(row["selectedMarketPrice"])
    assert row["bsmPrice"] == pytest.approx(row["selectedMarketPrice"], abs=1e-8)
    assert row["impliedVolContribution"] > 0.0
    assert abs(row["modelResidual"]) < 1e-8


def test_binomial_american_matches_european_call_without_dividends():
    params = dict(spot=100.0, strike=100.0, time_to_expiry=1.0, risk_free_rate=0.03, volatility=0.20)
    american = binomial_american_price(**params, option_type="call", dividend_yield=0.0, steps=250)
    european = BlackScholesModel.call_price(
        params["spot"],
        params["strike"],
        params["time_to_expiry"],
        params["risk_free_rate"],
        params["volatility"],
        0.0,
    )

    assert american == pytest.approx(european, abs=0.03)


def test_american_put_has_non_negative_early_exercise_premium():
    european = BlackScholesModel.put_price(100.0, 110.0, 1.0, 0.05, 0.20, 0.0)
    american = binomial_american_price(100.0, 110.0, 1.0, 0.05, 0.20, "put", steps=250)

    assert american >= european
    assert american - european > 0.0


def test_apply_american_pricing_adds_comparison_columns():
    priced = apply_american_pricing(_chain(), 100.0, steps=100)

    assert {"europeanPrice", "americanPrice", "earlyExercisePremium", "americanModel"}.issubset(priced.columns)
    assert priced["americanPrice"].notna().all()
    assert (priced["earlyExercisePremium"] >= 0.0).all()


def test_connector_surface_shock_metadata_is_dashboard_ready():
    metadata = DashboardConnector._surface_shock_metadata(_chain(), 100.0)

    assert metadata["surface_shock_available"] is True
    assert metadata["surface_shock_contracts"] == 3
    assert metadata["surface_shocks"]["scenarios"]
