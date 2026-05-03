from datetime import datetime, timedelta

import pandas as pd

from src.data.options_provider import YFinanceOptionsProvider


def test_yfinance_options_normalize_filters_and_shapes_rows():
    now = datetime(2026, 5, 2, 12, 0, 0)
    expiration = (now + timedelta(days=30)).strftime("%Y-%m-%d")
    raw = pd.DataFrame(
        [
            {
                "contractSymbol": "AAPL260601C00195000",
                "strike": 195.0,
                "lastPrice": 8.2,
                "bid": 8.0,
                "ask": 8.4,
                "volume": 120,
                "openInterest": 500,
                "impliedVolatility": 0.24,
                "type": "call",
                "expiration": expiration,
            },
            {
                "contractSymbol": "AAPL260601C00000000",
                "strike": 0.0,
                "lastPrice": 0.0,
                "bid": 0.0,
                "ask": 0.0,
                "volume": 0,
                "openInterest": 0,
                "impliedVolatility": 0.0,
                "type": "call",
                "expiration": expiration,
            },
        ]
    )

    clean = YFinanceOptionsProvider._normalize(raw, "AAPL", 200.0, now)

    assert len(clean) == 1
    row = clean.iloc[0]
    assert row["symbol"] == "AAPL"
    assert row["daysToExpiration"] == 30
    assert row["moneyness"] == 0.975
    assert row["mid"] == 8.2
    assert row["bidAskSpreadPct"] == (8.4 - 8.0) / 8.2
