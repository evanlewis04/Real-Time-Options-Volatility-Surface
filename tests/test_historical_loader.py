import pandas as pd

from src.data.historical import HistoricalPriceLoader


def test_historical_loader_returns_realized_vol_inputs():
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    frame = pd.DataFrame({"Close": [100 + i for i in range(80)]}, index=dates)
    loader = HistoricalPriceLoader(fetcher=lambda symbol, period: frame)

    result = loader.load("aapl", "6mo")

    assert result.available
    assert result.symbol == "AAPL"
    assert len(result.returns()) == 79
    assert result.realized_vol(20).notna().any()


def test_historical_loader_caches_fetches():
    calls = {"count": 0}

    def fetcher(symbol, period):
        calls["count"] += 1
        return pd.DataFrame({"Close": [100.0, 101.0, 102.0]})

    loader = HistoricalPriceLoader(fetcher=fetcher)
    first = loader.load("AAPL", "1mo")
    second = loader.load("AAPL", "1mo")

    assert first is second
    assert calls["count"] == 1


def test_historical_loader_returns_unavailable_result_on_empty_data():
    loader = HistoricalPriceLoader(fetcher=lambda symbol, period: pd.DataFrame())

    result = loader.load("AAPL", "1y")

    assert not result.available
    assert result.fallback_reason == "No historical closes returned"
