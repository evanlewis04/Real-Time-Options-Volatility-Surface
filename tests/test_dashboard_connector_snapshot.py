from datetime import datetime

import pandas as pd

from dashboard_connector import DashboardConnector
from src.data.market_calendar import EASTERN
from src.data.models import MarketDataSnapshot
from src.pricing.black_scholes import BlackScholesModel


class StubPriceProvider:
    yfinance_working = True

    def get_live_price(self, symbol: str) -> float:
        return 200.0


class StubOptionsProvider:
    def __init__(self):
        self.settings = {
            "min_open_interest": 0,
            "min_volume": 0,
            "max_bid_ask_spread_pct": 1.5,
            "max_quote_age_days": 5,
        }

    def configure_liquidity_filters(self, **kwargs):
        old = dict(self.settings)
        for key, value in kwargs.items():
            if value is not None:
                self.settings[key] = value
        return old != self.settings

    def liquidity_filter_settings(self):
        return dict(self.settings)

    def fetch_chain(self, symbol: str, spot_price: float):
        frame = pd.DataFrame(
            [
                {
                    "contractSymbol": f"{symbol}260619C00200000",
                    "type": "call",
                    "expiration": pd.Timestamp("2026-06-19"),
                    "daysToExpiration": 47,
                    "strike": spot_price,
                    "moneyness": 1.0,
                    "bid": 8.0,
                    "ask": 8.4,
                    "mid": 8.2,
                    "mark": 8.2,
                    "last": 8.1,
                    "volume": 120,
                    "openInterest": 500,
                    "impliedVolatility": 0.24,
                    "riskFreeRate": 0.051,
                    "dividendYield": 0.005,
                    "effectiveDividendYield": 0.015,
                    "discreteDividendAmount": 0.26,
                    "discreteDividendPV": 0.259,
                    "discreteDividendCount": 1,
                    "bidAskSpread": 0.4,
                    "bidAskSpreadPct": 0.04878,
                    "isCrossedMarket": False,
                    "isLockedMarket": False,
                }
            ]
        )
        meta = {
            "symbol": symbol,
            "source": "fixture",
            "mode": "Live/Delayed",
            "timestamp": datetime(2026, 5, 3, 10, 0, 0),
            "raw_rows": 1,
            "valid_rows": 1,
            "rejected_rows": 0,
            **self.settings,
            "liquidity_filtered_count": 0,
            "crossed_market_count": 0,
            "locked_market_count": 0,
            "crossed_locked_rejected_count": 0,
            "parity_pairs_checked": 0,
            "parity_violation_count": 0,
            "parity_violation_rows": 0,
            "parity_violations": [],
            "rejection_reasons": {},
        }
        return frame, meta


def test_connector_returns_canonical_market_data_snapshot(tmp_path):
    connector = DashboardConnector()
    connector.price_provider = StubPriceProvider()
    connector.options_provider = StubOptionsProvider()
    connector.snapshot_dir = tmp_path
    connector.chain_cache.clear()

    snapshot = connector.get_market_data_snapshot("aapl")

    assert isinstance(snapshot, MarketDataSnapshot)
    assert snapshot.symbol == "AAPL"
    assert snapshot.spot == 200.0
    assert snapshot.source == "fixture"
    assert len(snapshot.options) == 1
    assert snapshot.options[0].contract == "AAPL260619C00200000"
    assert snapshot.options[0].risk_free_rate is not None
    assert snapshot.options[0].discount_factor is not None
    assert snapshot.options[0].forward_price is not None
    assert snapshot.options[0].forward_moneyness is not None
    assert snapshot.options[0].log_moneyness is not None
    assert snapshot.options[0].effective_dividend_yield is not None
    assert snapshot.risk_free_rate_source is not None
    assert snapshot.expiry_rates
    assert snapshot.dividend_source is not None
    assert snapshot.expiry_dividends
    assert snapshot.corporate_action_source is not None
    assert snapshot.corporate_action_warning_count >= 1
    assert snapshot.expiry_corporate_actions
    assert snapshot.min_open_interest == 0
    assert snapshot.liquidity_filtered_count == 0
    assert snapshot.crossed_locked_rejected_count == 0
    assert snapshot.parity_violation_count == 0
    assert snapshot.data_quality_score == 100.0
    assert dict(snapshot.expiry_quality)["2026-06-19"]["valid_quotes"] == 1


def test_connector_options_chain_snapshot_uses_canonical_model_shape(tmp_path):
    connector = DashboardConnector()
    connector.price_provider = StubPriceProvider()
    connector.options_provider = StubOptionsProvider()
    connector.snapshot_dir = tmp_path
    connector.chain_cache.clear()

    frame, meta = connector.get_options_chain_snapshot("AAPL")

    assert frame.iloc[0]["contractSymbol"] == "AAPL260619C00200000"
    assert frame.iloc[0]["impliedVolatility"] == 0.24
    assert frame.iloc[0]["riskFreeRate"] > 0.0
    assert frame.iloc[0]["discountFactor"] > 0.0
    assert frame.iloc[0]["forwardPrice"] > 0.0
    assert frame.iloc[0]["forwardMoneyness"] > 0.0
    assert frame.iloc[0]["effectiveDividendYield"] >= 0.0
    assert frame.iloc[0]["selectedMarketPrice"] == 8.2
    assert frame.iloc[0]["selectedPriceSource"] == "mark"
    assert frame.iloc[0]["computedIV"] > 0.0
    assert meta["source"] == "fixture"
    assert meta["valid_rows"] == 1
    assert meta["option_price_source"] == "mark"
    assert meta["computed_iv_count"] == 1
    assert meta["risk_free_rate_30d"] > 0.0
    assert meta["forward_price_median"] > 0.0
    assert meta["discount_factor_median"] > 0.0
    assert "2026-06-19" in meta["expiry_forwards"]
    assert meta["expiry_rates"]["2026-06-19"] > 0.0
    assert meta["effective_dividend_yield_30d"] >= 0.0
    assert "2026-06-19" in meta["expiry_dividends"]
    assert meta["corporate_action_warning_count"] >= 1
    assert "2026-06-19" in meta["expiry_corporate_actions"]
    assert any("dividend" in warning for warning in meta["corporate_action_warnings"])
    assert meta["liquidity_filtered_count"] == 0
    assert meta["crossed_locked_rejected_count"] == 0
    assert meta["parity_violation_count"] == 0
    assert meta["rejection_reasons"] == {}
    assert meta["data_quality_score"] == 100.0
    assert meta["quality_reason_buckets"] == {}
    assert meta["expiry_quality"]["2026-06-19"]["valid_quotes"] == 1
    assert meta["expiry_quality"]["2026-06-19"]["score"] == 100.0


def test_connector_configures_liquidity_filters_and_clears_cache(tmp_path):
    connector = DashboardConnector()
    connector.price_provider = StubPriceProvider()
    connector.options_provider = StubOptionsProvider()
    connector.snapshot_dir = tmp_path
    connector.chain_cache["AAPL"] = (pd.DataFrame([{"x": 1}]), {}, datetime(2026, 5, 3))
    connector.surface_metadata["AAPL"] = {"valid_rows": 1}

    settings = connector.configure_liquidity_filters(
        min_open_interest=100,
        min_volume=25,
        max_bid_ask_spread_pct=0.40,
        max_quote_age_days=2,
    )

    assert settings["min_open_interest"] == 100
    assert settings["min_volume"] == 25
    assert settings["max_bid_ask_spread_pct"] == 0.40
    assert settings["max_quote_age_days"] == 2
    assert connector.chain_cache == {}
    assert connector.surface_metadata == {}


def test_connector_option_price_source_drives_computed_iv():
    connector = DashboardConnector()
    spot = 100.0
    strike = 100.0
    expiry = 30 / 365.0
    rate = 0.0
    mid_price = BlackScholesModel.call_price(spot, strike, expiry, rate, 0.20)
    last_price = BlackScholesModel.call_price(spot, strike, expiry, rate, 0.40)
    chain = pd.DataFrame(
        [
            {
                "type": "call",
                "strike": strike,
                "time_to_expiry": expiry,
                "riskFreeRate": rate,
                "effectiveDividendYield": 0.0,
                "bid": mid_price - 0.05,
                "ask": mid_price + 0.05,
                "mid": mid_price,
                "mark": mid_price,
                "last": last_price,
            }
        ]
    )

    connector.configure_option_price_source("midpoint")
    midpoint_chain, midpoint_meta = connector._apply_option_price_source(chain, spot)
    connector.configure_option_price_source("last")
    last_chain, last_meta = connector._apply_option_price_source(chain, spot)

    assert midpoint_chain.iloc[0]["selectedMarketPrice"] == mid_price
    assert midpoint_chain.iloc[0]["selectedPriceSource"] == "midpoint"
    assert abs(midpoint_chain.iloc[0]["computedIV"] - 0.20) < 1e-4
    assert midpoint_meta["computed_iv_count"] == 1
    assert last_chain.iloc[0]["selectedMarketPrice"] == last_price
    assert last_chain.iloc[0]["selectedPriceSource"] == "last"
    assert abs(last_chain.iloc[0]["computedIV"] - 0.40) < 1e-4
    assert last_meta["option_price_source"] == "last"


def test_connector_flags_obvious_put_call_parity_violations():
    connector = DashboardConnector()
    expiry = pd.Timestamp("2026-06-19")
    chain = pd.DataFrame(
        [
            {
                "type": "call",
                "expiration": expiry,
                "daysToExpiration": 30,
                "strike": 100.0,
                "selectedMarketPrice": 12.0,
                "riskFreeRate": 0.0,
                "effectiveDividendYield": 0.0,
            },
            {
                "type": "put",
                "expiration": expiry,
                "daysToExpiration": 30,
                "strike": 100.0,
                "selectedMarketPrice": 2.0,
                "riskFreeRate": 0.0,
                "effectiveDividendYield": 0.0,
            },
            {
                "type": "call",
                "expiration": expiry,
                "daysToExpiration": 30,
                "strike": 110.0,
                "selectedMarketPrice": 2.0,
                "riskFreeRate": 0.0,
                "effectiveDividendYield": 0.0,
            },
            {
                "type": "put",
                "expiration": expiry,
                "daysToExpiration": 30,
                "strike": 110.0,
                "selectedMarketPrice": 12.0,
                "riskFreeRate": 0.0,
                "effectiveDividendYield": 0.0,
            },
        ]
    )

    checked, meta = connector._apply_parity_checks(chain, 100.0)

    assert meta["parity_pairs_checked"] == 2
    assert meta["parity_violation_count"] == 1
    assert meta["parity_violation_rows"] == 2
    assert checked.loc[checked["strike"] == 100.0, "parityViolation"].all()
    assert not checked.loc[checked["strike"] == 110.0, "parityViolation"].any()
    assert meta["parity_violations"][0]["strike"] == 100.0


def test_connector_quality_score_penalizes_rejections_computed_iv_and_parity():
    connector = DashboardConnector()
    expiry = pd.Timestamp("2026-06-19")
    chain = pd.DataFrame(
        [
            {
                "type": "call",
                "expiration": expiry,
                "computedIV": 0.20,
                "parityViolation": False,
            },
            {
                "type": "put",
                "expiration": expiry,
                "computedIV": None,
                "parityViolation": True,
            },
        ]
    )
    metadata = {
        "valid_rows": 2,
        "rejected_rows": 2,
        "rejection_reasons": {"low_volume": 2},
        "computed_iv_failed_count": 1,
        "parity_violation_rows": 1,
        "expiry_quality": {
            "2026-06-19": {
                "raw_quotes": 4,
                "valid_quotes": 2,
                "rejected_quotes": 2,
                "reason_buckets": {"low_volume": 2},
            }
        },
    }

    quality = connector._data_quality_metadata(chain, metadata)

    assert quality["quality_reason_buckets"] == {
        "low_volume": 2,
        "computed_iv_failed": 1,
        "parity_violation": 1,
    }
    assert quality["data_quality_score"] == 32.5
    assert quality["expiry_quality"]["2026-06-19"]["reason_buckets"] == {
        "low_volume": 2,
        "computed_iv_failed": 1,
        "parity_violation": 1,
    }
    assert quality["expiry_quality"]["2026-06-19"]["score"] == 32.5


def test_connector_quality_score_penalizes_no_arbitrage_violations():
    connector = DashboardConnector()
    expiry = pd.Timestamp("2026-06-19")
    chain = pd.DataFrame(
        [
            {"type": "call", "expiration": expiry, "computedIV": 0.20, "noArbitrageViolation": False},
            {"type": "call", "expiration": expiry, "computedIV": 0.21, "noArbitrageViolation": True},
        ]
    )
    metadata = {
        "valid_rows": 2,
        "rejected_rows": 0,
        "no_arbitrage_violation_rows": 1,
        "expiry_quality": {"2026-06-19": {"valid_quotes": 2, "rejected_quotes": 0, "reason_buckets": {}}},
    }

    quality = connector._data_quality_metadata(chain, metadata)

    assert quality["quality_reason_buckets"] == {"no_arbitrage_violation": 1}
    assert quality["data_quality_score"] == 90.0
    assert quality["expiry_quality"]["2026-06-19"]["reason_buckets"] == {"no_arbitrage_violation": 1}


def test_connector_exposes_market_calendar_status():
    connector = DashboardConnector()

    status = connector.market_calendar.status(datetime(2026, 5, 4, 10, 0, tzinfo=EASTERN)).as_dict()

    assert status["session_state"] == "Open"
    assert status["data_delay_minutes"] == 15
