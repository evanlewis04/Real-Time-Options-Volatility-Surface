from datetime import datetime, timedelta

import pandas as pd

from src.data.models import MarketDataSnapshot, option_quotes_from_frame
from src.data.snapshots import list_snapshots, load_latest_snapshot, load_snapshot, save_snapshot


def _snapshot() -> MarketDataSnapshot:
    frame = pd.DataFrame(
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
                "selectedMarketPrice": 8.2,
                "selectedPriceSource": "mark",
                "ivInput": "computed",
                "parityViolation": False,
                "parityError": 0.05,
                "parityTheoreticalDiff": 0.10,
                "parityObservedDiff": 0.15,
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
            }
        ]
    )
    quotes = tuple(option_quotes_from_frame(frame))
    return MarketDataSnapshot(
        symbol="AAPL",
        spot=200.0,
        spot_timestamp=datetime(2026, 5, 3, 10, 0, 0),
        chain_timestamp=datetime(2026, 5, 3, 10, 0, 2),
        expirations=(datetime(2026, 6, 19),),
        options=quotes,
        source="fixture",
        source_delay=timedelta(minutes=15),
        cache_age=timedelta(seconds=7),
        mode="Live/Delayed",
        risk_free_rate_source="local:config/risk_free_curve.csv",
        risk_free_rate_mode="Local",
        risk_free_rate_timestamp=datetime(2026, 5, 3, 9, 59, 0),
        risk_free_rate_curve=((30, 0.051), (90, 0.0495)),
        expiry_rates=(("2026-06-19", 0.0505),),
        risk_free_rate_30d=0.051,
        risk_free_rate_min=0.0505,
        risk_free_rate_max=0.0505,
        risk_free_rate_median=0.0505,
        dividend_source="local:config/dividends.csv",
        dividend_mode="Local",
        dividend_timestamp=datetime(2026, 5, 3, 9, 58, 0),
        annual_dividend_yield=0.005,
        dividend_events=({"ex_date": "2026-05-15", "amount": 0.26, "currency": "USD"},),
        expiry_dividends=(
            (
                "2026-06-19",
                {
                    "annual_yield": 0.005,
                    "effective_yield": 0.015,
                    "discrete_amount": 0.26,
                    "discrete_present_value": 0.259,
                    "discrete_count": 1,
                },
            ),
        ),
        effective_dividend_yield_30d=0.012,
        effective_dividend_yield_min=0.005,
        effective_dividend_yield_max=0.015,
        effective_dividend_yield_median=0.015,
        corporate_action_source="local:config/corporate_actions.csv",
        corporate_action_mode="Local",
        corporate_action_timestamp=datetime(2026, 5, 3, 9, 57, 0),
        corporate_actions=(
            {
                "symbol": "AAPL",
                "action_type": "dividend",
                "effective_date": "2026-05-15",
                "description": "Cash dividend",
                "value": 0.26,
                "ratio": None,
                "source": "fixture",
            },
        ),
        upcoming_corporate_actions=(
            {
                "symbol": "AAPL",
                "action_type": "dividend",
                "effective_date": "2026-05-15",
                "description": "Cash dividend",
                "value": 0.26,
                "ratio": None,
                "source": "fixture",
            },
        ),
        expiry_corporate_actions=(
            (
                "2026-06-19",
                [
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
            ),
        ),
        corporate_action_warning_count=1,
        corporate_action_warnings=("AAPL dividend on 2026-05-15: Cash dividend (0.26)",),
        stale_quote_count=0,
        last_only_quote_count=0,
        zero_bid_ask_count=0,
        crossed_market_count=1,
        locked_market_count=1,
        crossed_locked_rejected_count=2,
        stale_last_only_rejected_count=1,
        min_open_interest=100,
        min_volume=10,
        max_bid_ask_spread_pct=0.5,
        liquidity_filtered_count=2,
        low_open_interest_rejected_count=1,
        low_volume_rejected_count=1,
        rejection_reasons=(("crossed_locked_market", 2), ("low_open_interest", 1), ("low_volume", 1)),
        max_quote_age_days=5,
        option_price_source="mark",
        computed_iv_count=1,
        computed_iv_failed_count=0,
        parity_pairs_checked=1,
        parity_violation_count=1,
        parity_violation_rows=2,
        parity_violations=(
            {
                "expiration": "2026-06-19",
                "strike": 200.0,
                "parity_error": 2.5,
            },
        ),
        raw_rows=1,
        valid_rows=1,
    )


def test_save_and_load_snapshot_round_trips_options_and_metadata(tmp_path):
    metadata_path = save_snapshot(_snapshot(), tmp_path)

    loaded = load_snapshot(metadata_path)

    assert loaded.symbol == "AAPL"
    assert loaded.spot == 200.0
    assert loaded.source_delay == timedelta(minutes=15)
    assert loaded.cache_age == timedelta(seconds=7)
    assert len(loaded.options) == 1
    assert loaded.options[0].contract == "AAPL260619C00200000"
    assert loaded.options[0].risk_free_rate == 0.051
    assert loaded.risk_free_rate_source == "local:config/risk_free_curve.csv"
    assert loaded.expiry_rates == (("2026-06-19", 0.0505),)
    assert loaded.risk_free_rate_30d == 0.051
    assert loaded.options[0].effective_dividend_yield == 0.015
    assert loaded.annual_dividend_yield == 0.005
    assert loaded.expiry_dividends[0][1]["discrete_count"] == 1
    assert loaded.corporate_action_warning_count == 1
    assert loaded.expiry_corporate_actions[0][1][0]["action_type"] == "dividend"
    assert loaded.options[0].quote_quality == "bid_ask"
    assert loaded.options[0].is_crossed_market is False
    assert loaded.options[0].is_locked_market is False
    assert loaded.options[0].mark == 8.2
    assert loaded.options[0].computed_iv == 0.241
    assert loaded.options[0].selected_market_price == 8.2
    assert loaded.options[0].parity_violation is False
    assert loaded.options[0].parity_error == 0.05
    assert loaded.stale_last_only_rejected_count == 1
    assert loaded.min_open_interest == 100
    assert loaded.min_volume == 10
    assert loaded.max_bid_ask_spread_pct == 0.5
    assert loaded.liquidity_filtered_count == 2
    assert loaded.crossed_market_count == 1
    assert loaded.locked_market_count == 1
    assert loaded.crossed_locked_rejected_count == 2
    assert loaded.rejection_reasons == (
        ("crossed_locked_market", 2),
        ("low_open_interest", 1),
        ("low_volume", 1),
    )
    assert loaded.max_quote_age_days == 5
    assert loaded.option_price_source == "mark"
    assert loaded.computed_iv_count == 1
    assert loaded.computed_iv_failed_count == 0
    assert loaded.parity_pairs_checked == 1
    assert loaded.parity_violation_count == 1
    assert loaded.parity_violation_rows == 2
    assert loaded.parity_violations[0]["strike"] == 200.0


def test_list_and_load_latest_snapshot(tmp_path):
    metadata_path = save_snapshot(_snapshot(), tmp_path)

    assert list_snapshots("AAPL", tmp_path) == [metadata_path]
    assert load_latest_snapshot("AAPL", tmp_path).symbol == "AAPL"
    assert load_latest_snapshot("MSFT", tmp_path) is None
