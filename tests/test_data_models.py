from datetime import datetime, timedelta

import pandas as pd

from src.data.models import MarketDataSnapshot, OptionQuote, option_quotes_from_frame, option_quotes_to_frame


def _sample_chain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contractSymbol": "AAPL260619C00200000",
                "type": "call",
                "expiration": pd.Timestamp("2026-06-19"),
                "daysToExpiration": 47,
                "strike": 200.0,
                "moneyness": 1.0,
                "bid": 8.0,
                "ask": 8.4,
                "mid": 8.2,
                "mark": 8.2,
                "last": 8.1,
                "volume": 120,
                "openInterest": 500,
                "impliedVolatility": 0.24,
                "computedIV": 0.241,
                "vanna": 0.0008,
                "volga": 0.0012,
                "vomma": 0.0012,
                "charm": -0.002,
                "speed": -0.0001,
                "color": 0.0003,
                "selectedMarketPrice": 8.2,
                "selectedPriceSource": "mark",
                "ivInput": "computed",
                "parityViolation": False,
                "parityError": 0.05,
                "parityTheoreticalDiff": 0.10,
                "parityObservedDiff": 0.15,
                "noArbitrageViolation": False,
                "noArbitrageReasons": "",
                "noArbitrageLowerBound": 0.0,
                "noArbitrageUpperBound": 199.0,
                "noArbitrageBoundViolation": False,
                "noArbitrageMonotonicityViolation": False,
                "noArbitrageConvexityViolation": False,
                "noArbitrageCalendarViolation": False,
                "riskFreeRate": 0.051,
                "dividendYield": 0.005,
                "effectiveDividendYield": 0.015,
                "discreteDividendAmount": 0.26,
                "discreteDividendPV": 0.259,
                "discreteDividendCount": 1,
                "quoteQuality": "bid_ask",
                "isStaleQuote": False,
                "isCrossedMarket": False,
                "isLockedMarket": False,
                "quoteAgeSeconds": 3600,
                "bidAskSpread": 0.4,
                "bidAskSpreadPct": 0.04878,
                "quoteTimestamp": pd.Timestamp("2026-05-03 10:00:00"),
            }
        ]
    )


def test_option_quote_from_frame_round_trips_dashboard_shape():
    quotes = option_quotes_from_frame(_sample_chain())

    assert len(quotes) == 1
    quote = quotes[0]
    assert isinstance(quote, OptionQuote)
    assert quote.contract == "AAPL260619C00200000"
    assert quote.type == "call"
    assert quote.raw_iv == 0.24
    assert quote.computed_iv == 0.241
    assert quote.vanna == 0.0008
    assert quote.vomma == 0.0012
    assert quote.charm == -0.002
    assert quote.mark == 8.2
    assert quote.selected_market_price == 8.2
    assert quote.selected_price_source == "mark"
    assert quote.iv_input == "computed"
    assert quote.parity_violation is False
    assert quote.parity_error == 0.05
    assert quote.no_arbitrage_violation is False
    assert quote.no_arbitrage_upper_bound == 199.0
    assert quote.risk_free_rate == 0.051
    assert quote.effective_dividend_yield == 0.015
    assert quote.discrete_dividend_count == 1
    assert quote.quote_quality == "bid_ask"
    assert quote.is_stale_quote is False
    assert quote.is_crossed_market is False
    assert quote.is_locked_market is False
    assert quote.quote_age_seconds == 3600
    assert quote.open_interest == 500

    frame = option_quotes_to_frame(quotes)
    assert frame.iloc[0]["contractSymbol"] == quote.contract
    assert frame.iloc[0]["impliedVolatility"] == quote.raw_iv
    assert frame.iloc[0]["computedIV"] == quote.computed_iv
    assert frame.iloc[0]["vanna"] == quote.vanna
    assert frame.iloc[0]["volga"] == quote.volga
    assert frame.iloc[0]["color"] == quote.color
    assert frame.iloc[0]["selectedMarketPrice"] == quote.selected_market_price
    assert not frame.iloc[0]["parityViolation"]
    assert frame.iloc[0]["parityError"] == 0.05
    assert not frame.iloc[0]["noArbitrageViolation"]
    assert frame.iloc[0]["noArbitrageUpperBound"] == 199.0
    assert frame.iloc[0]["riskFreeRate"] == quote.risk_free_rate
    assert frame.iloc[0]["effectiveDividendYield"] == quote.effective_dividend_yield
    assert frame.iloc[0]["quoteQuality"] == "bid_ask"
    assert not frame.iloc[0]["isCrossedMarket"]
    assert not frame.iloc[0]["isLockedMarket"]
    assert frame.iloc[0]["time_to_expiry"] == quote.dte / 365.0


def test_market_data_snapshot_from_chain_frame_carries_metadata():
    now = datetime(2026, 5, 3, 10, 1, 0)
    snapshot = MarketDataSnapshot.from_chain_frame(
        "aapl",
        200.0,
        now,
        _sample_chain(),
        {
            "source": "yfinance",
            "mode": "Live/Delayed",
            "timestamp": now,
            "raw_rows": 2,
            "valid_rows": 1,
            "rejected_rows": 1,
            "cache_age_seconds": 12,
            "fallback_reason": None,
            "warnings": ["one row rejected"],
            "risk_free_rate_source": "local:config/risk_free_curve.csv",
            "risk_free_rate_mode": "Local",
            "risk_free_rate_timestamp": now,
            "risk_free_rate_curve": [{"tenor_days": 30, "rate": 0.051}],
            "expiry_rates": {"2026-06-19": 0.0505},
            "risk_free_rate_30d": 0.051,
            "risk_free_rate_min": 0.0505,
            "risk_free_rate_max": 0.0505,
            "risk_free_rate_median": 0.0505,
            "dividend_source": "local:config/dividends.csv",
            "dividend_mode": "Local",
            "dividend_timestamp": now,
            "annual_dividend_yield": 0.005,
            "dividend_events": [{"ex_date": "2026-05-15", "amount": 0.26, "currency": "USD"}],
            "expiry_dividends": {
                "2026-06-19": {
                    "annual_yield": 0.005,
                    "effective_yield": 0.015,
                    "discrete_amount": 0.26,
                    "discrete_present_value": 0.259,
                    "discrete_count": 1,
                }
            },
            "effective_dividend_yield_30d": 0.012,
            "effective_dividend_yield_min": 0.005,
            "effective_dividend_yield_max": 0.015,
            "effective_dividend_yield_median": 0.015,
            "corporate_action_source": "local:config/corporate_actions.csv",
            "corporate_action_mode": "Local",
            "corporate_action_timestamp": now,
            "corporate_actions": [
                {
                    "symbol": "AAPL",
                    "action_type": "dividend",
                    "effective_date": "2026-05-15",
                    "description": "Cash dividend",
                    "value": 0.26,
                    "ratio": None,
                    "source": "fixture",
                }
            ],
            "upcoming_corporate_actions": [
                {
                    "symbol": "AAPL",
                    "action_type": "dividend",
                    "effective_date": "2026-05-15",
                    "description": "Cash dividend",
                    "value": 0.26,
                    "ratio": None,
                    "source": "fixture",
                }
            ],
            "expiry_corporate_actions": {
                "2026-06-19": [
                    {
                        "symbol": "AAPL",
                        "action_type": "dividend",
                        "effective_date": "2026-05-15",
                        "description": "Cash dividend",
                        "value": 0.26,
                        "ratio": None,
                        "source": "fixture",
                    }
                ]
            },
            "corporate_action_warning_count": 1,
            "corporate_action_warnings": ["AAPL dividend on 2026-05-15: Cash dividend (0.26)"],
            "stale_quote_count": 0,
            "last_only_quote_count": 0,
            "zero_bid_ask_count": 0,
            "crossed_market_count": 1,
            "locked_market_count": 1,
            "crossed_locked_rejected_count": 2,
            "stale_last_only_rejected_count": 1,
            "min_open_interest": 100,
            "min_volume": 10,
            "max_bid_ask_spread_pct": 0.50,
            "liquidity_filtered_count": 3,
            "low_open_interest_rejected_count": 1,
            "low_volume_rejected_count": 1,
            "wide_spread_rejected_count": 1,
            "old_quote_rejected_count": 0,
            "rejection_reasons": {
                "low_open_interest": 1,
                "low_volume": 1,
                "wide_bid_ask_spread": 1,
                "crossed_locked_market": 2,
            },
            "max_quote_age_days": 5,
            "option_price_source": "mark",
            "computed_iv_count": 1,
            "computed_iv_failed_count": 0,
            "parity_pairs_checked": 1,
            "parity_violation_count": 1,
            "parity_violation_rows": 2,
            "parity_violations": [
                {
                    "expiration": "2026-06-19",
                    "strike": 200.0,
                    "parity_error": 2.5,
                }
            ],
            "no_arbitrage_checks": ["bounds_by_type", "call_monotonicity"],
            "no_arbitrage_violation_count": 1,
            "no_arbitrage_violation_rows": 1,
            "no_arbitrage_reason_buckets": {"bounds": 1},
            "no_arbitrage_violations": [
                {
                    "check": "bounds",
                    "expiration": "2026-06-19",
                    "strike": 200.0,
                }
            ],
            "no_arbitrage_excluded_count": 1,
        },
    )

    assert snapshot.symbol == "AAPL"
    assert snapshot.spot == 200.0
    assert snapshot.source == "yfinance"
    assert snapshot.source_delay == timedelta(minutes=15)
    assert snapshot.cache_age == timedelta(seconds=12)
    assert snapshot.valid_rows == 1
    assert snapshot.rejected_rows == 1
    assert snapshot.expirations == (datetime(2026, 6, 19),)
    assert snapshot.metadata_dict()["cache_age_seconds"] == 12
    assert snapshot.options[0].risk_free_rate == 0.051
    assert snapshot.metadata_dict()["risk_free_rate_source"] == "local:config/risk_free_curve.csv"
    assert snapshot.metadata_dict()["expiry_rates"]["2026-06-19"] == 0.0505
    assert snapshot.metadata_dict()["risk_free_rate_30d"] == 0.051
    assert snapshot.metadata_dict()["annual_dividend_yield"] == 0.005
    assert snapshot.metadata_dict()["expiry_dividends"]["2026-06-19"]["discrete_count"] == 1
    assert snapshot.metadata_dict()["corporate_action_warning_count"] == 1
    assert snapshot.metadata_dict()["expiry_corporate_actions"]["2026-06-19"][0]["action_type"] == "dividend"
    assert snapshot.metadata_dict()["crossed_market_count"] == 1
    assert snapshot.metadata_dict()["locked_market_count"] == 1
    assert snapshot.metadata_dict()["crossed_locked_rejected_count"] == 2
    assert snapshot.metadata_dict()["stale_last_only_rejected_count"] == 1
    assert snapshot.metadata_dict()["min_open_interest"] == 100
    assert snapshot.metadata_dict()["min_volume"] == 10
    assert snapshot.metadata_dict()["max_bid_ask_spread_pct"] == 0.50
    assert snapshot.metadata_dict()["liquidity_filtered_count"] == 3
    assert snapshot.metadata_dict()["rejection_reasons"]["low_volume"] == 1
    assert snapshot.metadata_dict()["rejection_reasons"]["crossed_locked_market"] == 2
    assert snapshot.metadata_dict()["max_quote_age_days"] == 5
    assert snapshot.metadata_dict()["option_price_source"] == "mark"
    assert snapshot.metadata_dict()["computed_iv_count"] == 1
    assert snapshot.metadata_dict()["computed_iv_failed_count"] == 0
    assert snapshot.metadata_dict()["parity_pairs_checked"] == 1
    assert snapshot.metadata_dict()["parity_violation_count"] == 1
    assert snapshot.metadata_dict()["parity_violation_rows"] == 2
    assert snapshot.metadata_dict()["parity_violations"][0]["strike"] == 200.0
    assert snapshot.metadata_dict()["no_arbitrage_checks"] == ["bounds_by_type", "call_monotonicity"]
    assert snapshot.metadata_dict()["no_arbitrage_violation_count"] == 1
    assert snapshot.metadata_dict()["no_arbitrage_violation_rows"] == 1
    assert snapshot.metadata_dict()["no_arbitrage_reason_buckets"]["bounds"] == 1
    assert snapshot.metadata_dict()["no_arbitrage_violations"][0]["check"] == "bounds"
    assert snapshot.metadata_dict()["no_arbitrage_excluded_count"] == 1


def test_empty_snapshot_returns_empty_options_frame():
    snapshot = MarketDataSnapshot(
        symbol="XYZ",
        spot=100.0,
        spot_timestamp=datetime(2026, 5, 3),
        chain_timestamp=None,
        source="fixture",
        mode="Unavailable",
    )

    assert snapshot.options_frame().empty
    assert snapshot.metadata_dict()["mode"] == "Unavailable"
