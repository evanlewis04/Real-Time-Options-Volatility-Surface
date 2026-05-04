from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

import src.data.options_provider as options_provider_module
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
                "lastTradeDate": now - timedelta(hours=1),
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
    assert row["quoteQuality"] == "bid_ask"
    assert row["isStaleQuote"] == False
    assert row["quoteAgeSeconds"] == 3600.0


def test_yfinance_options_normalize_marks_stale_and_last_only_quotes():
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
                "lastTradeDate": now - timedelta(days=10),
                "type": "call",
                "expiration": expiration,
            },
            {
                "contractSymbol": "AAPL260601P00195000",
                "strike": 195.0,
                "lastPrice": 7.5,
                "bid": None,
                "ask": None,
                "volume": 80,
                "openInterest": 400,
                "impliedVolatility": 0.25,
                "lastTradeDate": now - timedelta(days=1),
                "type": "put",
                "expiration": expiration,
            },
            {
                "contractSymbol": "AAPL260601C00200000",
                "strike": 200.0,
                "lastPrice": 6.1,
                "bid": None,
                "ask": None,
                "volume": 10,
                "openInterest": 100,
                "impliedVolatility": 0.26,
                "lastTradeDate": now - timedelta(days=10),
                "type": "call",
                "expiration": expiration,
            },
        ]
    )

    clean = YFinanceOptionsProvider._normalize(raw, "AAPL", 200.0, now, max_quote_age_days=5)

    assert clean["contractSymbol"].tolist() == ["AAPL260601C00195000", "AAPL260601P00195000"]
    assert clean.iloc[0]["quoteQuality"] == "stale_bid_ask"
    assert clean.iloc[0]["isStaleQuote"] == True
    assert clean.iloc[1]["quoteQuality"] == "last_only"
    assert clean.attrs["stale_quote_count"] == 1
    assert clean.attrs["last_only_quote_count"] == 1
    assert clean.attrs["stale_last_only_rejected_count"] == 1


def test_yfinance_options_provider_caches_by_symbol_and_expiry(monkeypatch):
    calls = pd.DataFrame(
        [
            {
                "contractSymbol": "AAPL990619C00200000",
                "strike": 200.0,
                "lastPrice": 8.2,
                "bid": 8.0,
                "ask": 8.4,
                "volume": 120,
                "openInterest": 500,
                "impliedVolatility": 0.24,
            }
        ]
    )
    puts = pd.DataFrame(
        [
            {
                "contractSymbol": "AAPL990619P00200000",
                "strike": 200.0,
                "lastPrice": 7.8,
                "bid": 7.6,
                "ask": 8.0,
                "volume": 100,
                "openInterest": 450,
                "impliedVolatility": 0.25,
            }
        ]
    )

    class FakeTicker:
        calls = 0
        options = ["2099-06-19"]

        def __init__(self, symbol):
            self.symbol = symbol

        def option_chain(self, expiration):
            FakeTicker.calls += 1
            return SimpleNamespace(calls=calls, puts=puts)

    monkeypatch.setattr(options_provider_module, "YFINANCE_AVAILABLE", True)
    monkeypatch.setattr(options_provider_module, "yf", SimpleNamespace(Ticker=FakeTicker))

    provider = YFinanceOptionsProvider(max_expirations=1, cache_ttl_seconds=300)
    first, first_meta = provider.fetch_chain("AAPL", 200.0)
    second, second_meta = provider.fetch_chain("AAPL", 200.0)

    assert FakeTicker.calls == 1
    assert len(first) == 2
    assert len(second) == 2
    assert first_meta.expirations_loaded == 1
    assert second_meta.expirations_loaded == 1
    assert provider.cache_status()["entries"] == 1
