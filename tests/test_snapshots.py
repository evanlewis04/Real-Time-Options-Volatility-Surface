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
                "last": 8.1,
                "volume": 120,
                "openInterest": 500,
                "impliedVolatility": 0.24,
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


def test_list_and_load_latest_snapshot(tmp_path):
    metadata_path = save_snapshot(_snapshot(), tmp_path)

    assert list_snapshots("AAPL", tmp_path) == [metadata_path]
    assert load_latest_snapshot("AAPL", tmp_path).symbol == "AAPL"
    assert load_latest_snapshot("MSFT", tmp_path) is None
