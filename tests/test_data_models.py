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
                "last": 8.1,
                "volume": 120,
                "openInterest": 500,
                "impliedVolatility": 0.24,
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
    assert quote.open_interest == 500

    frame = option_quotes_to_frame(quotes)
    assert frame.iloc[0]["contractSymbol"] == quote.contract
    assert frame.iloc[0]["impliedVolatility"] == quote.raw_iv
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
