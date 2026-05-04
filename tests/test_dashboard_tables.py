from datetime import date

import pandas as pd

from src.dashboard.tables import dataframe_to_csv_bytes, filter_market_snapshot, filter_option_chain


def test_filter_market_snapshot_by_mode_and_min_iv():
    frame = pd.DataFrame(
        [
            {"Symbol": "AAPL", "Mode": "Live/Delayed", "30D IV": 0.22},
            {"Symbol": "XYZ", "Mode": "Fallback", "30D IV": 0.60},
            {"Symbol": "SPY", "Mode": "Live/Delayed", "30D IV": 0.12},
        ]
    )

    filtered = filter_market_snapshot(frame, modes=["Live/Delayed"], min_iv_30d=0.20)

    assert filtered["Symbol"].tolist() == ["AAPL"]


def test_filter_option_chain_applies_quality_and_user_filters():
    frame = pd.DataFrame(
        [
            {
                "type": "call",
                "expiration": pd.Timestamp("2026-06-19"),
                "strike": 100.0,
                "moneyness": 1.0,
                "openInterest": 200,
                "volume": 50,
                "impliedVolatility": 0.25,
                "bidAskSpreadPct": 0.10,
                "quoteAgeSeconds": 3600,
            },
            {
                "type": "put",
                "expiration": pd.Timestamp("2026-07-17"),
                "strike": 140.0,
                "moneyness": 1.4,
                "openInterest": 10,
                "volume": 2,
                "impliedVolatility": 0.80,
                "bidAskSpreadPct": 0.90,
                "quoteAgeSeconds": 10 * 24 * 60 * 60,
            },
        ]
    )

    filtered = filter_option_chain(
        frame,
        max_spread_pct=0.50,
        min_open_interest=100,
        min_volume=10,
        max_quote_age_days=5,
        option_types=["call"],
        expirations=[date(2026, 6, 19)],
        moneyness_range=(0.8, 1.2),
        iv_range=(0.1, 0.5),
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["type"] == "call"


def test_dataframe_to_csv_bytes_uses_utf8_csv():
    data = dataframe_to_csv_bytes(pd.DataFrame([{"Symbol": "AAPL", "Spot": 200.0}]))

    assert data.startswith(b"Symbol,Spot")
    assert b"AAPL,200.0" in data
