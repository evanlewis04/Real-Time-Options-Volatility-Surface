import pandas as pd

from dashboard_connector import DashboardConnector
from src.quant.arbitrage import apply_no_arbitrage_checks


def test_no_arbitrage_checks_flag_bounds_monotonicity_and_convexity():
    expiry = pd.Timestamp("2026-06-19")
    chain = pd.DataFrame(
        [
            {
                "type": "call",
                "expiration": expiry,
                "daysToExpiration": 30,
                "strike": 90.0,
                "selectedMarketPrice": 9.0,
                "computedIV": 0.20,
                "riskFreeRate": 0.0,
                "effectiveDividendYield": 0.0,
            },
            {
                "type": "call",
                "expiration": expiry,
                "daysToExpiration": 30,
                "strike": 100.0,
                "selectedMarketPrice": 18.0,
                "computedIV": 0.20,
                "riskFreeRate": 0.0,
                "effectiveDividendYield": 0.0,
            },
            {
                "type": "call",
                "expiration": expiry,
                "daysToExpiration": 30,
                "strike": 110.0,
                "selectedMarketPrice": 2.0,
                "computedIV": 0.20,
                "riskFreeRate": 0.0,
                "effectiveDividendYield": 0.0,
            },
        ]
    )

    checked, meta = apply_no_arbitrage_checks(chain, spot=100.0)

    assert checked["noArbitrageViolation"].tolist() == [True, True, False]
    assert checked.iloc[0]["noArbitrageBoundViolation"]
    assert checked.iloc[1]["noArbitrageMonotonicityViolation"]
    assert checked.iloc[1]["noArbitrageConvexityViolation"]
    assert meta["no_arbitrage_reason_buckets"] == {
        "bounds": 1,
        "call_monotonicity": 1,
        "convexity": 1,
    }


def test_no_arbitrage_checks_flag_calendar_total_variance_decrease():
    chain = pd.DataFrame(
        [
            {
                "type": "put",
                "expiration": pd.Timestamp("2026-06-19"),
                "daysToExpiration": 30,
                "strike": 100.0,
                "selectedMarketPrice": 2.0,
                "computedIV": 0.40,
                "riskFreeRate": 0.0,
                "effectiveDividendYield": 0.0,
            },
            {
                "type": "put",
                "expiration": pd.Timestamp("2026-07-17"),
                "daysToExpiration": 60,
                "strike": 100.0,
                "selectedMarketPrice": 3.0,
                "computedIV": 0.20,
                "riskFreeRate": 0.0,
                "effectiveDividendYield": 0.0,
            },
        ]
    )

    checked, meta = apply_no_arbitrage_checks(chain, spot=100.0)

    assert checked["noArbitrageCalendarViolation"].all()
    assert meta["no_arbitrage_reason_buckets"] == {"calendar_monotonicity": 1}


def test_connector_surface_chain_excludes_no_arbitrage_violations():
    chain = pd.DataFrame(
        [
            {"computedIV": 0.20, "noArbitrageViolation": False},
            {"computedIV": 0.21, "noArbitrageViolation": True},
            {"computedIV": None, "noArbitrageViolation": False},
        ]
    )

    surface_chain = DashboardConnector._surface_iv_chain(chain)

    assert len(surface_chain) == 1
    assert surface_chain.attrs["no_arbitrage_excluded_count"] == 1
