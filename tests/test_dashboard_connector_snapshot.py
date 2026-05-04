from datetime import datetime

import pandas as pd

from dashboard_connector import DashboardConnector
from src.data.market_calendar import EASTERN
from src.data.models import MarketDataSnapshot


class StubPriceProvider:
    yfinance_working = True

    def get_live_price(self, symbol: str) -> float:
        return 200.0


class StubOptionsProvider:
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
                    "last": 8.1,
                    "volume": 120,
                    "openInterest": 500,
                    "impliedVolatility": 0.24,
                    "bidAskSpread": 0.4,
                    "bidAskSpreadPct": 0.04878,
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


def test_connector_options_chain_snapshot_uses_canonical_model_shape(tmp_path):
    connector = DashboardConnector()
    connector.price_provider = StubPriceProvider()
    connector.options_provider = StubOptionsProvider()
    connector.snapshot_dir = tmp_path
    connector.chain_cache.clear()

    frame, meta = connector.get_options_chain_snapshot("AAPL")

    assert frame.iloc[0]["contractSymbol"] == "AAPL260619C00200000"
    assert frame.iloc[0]["impliedVolatility"] == 0.24
    assert meta["source"] == "fixture"
    assert meta["valid_rows"] == 1


def test_connector_exposes_market_calendar_status():
    connector = DashboardConnector()

    status = connector.market_calendar.status(datetime(2026, 5, 4, 10, 0, tzinfo=EASTERN)).as_dict()

    assert status["session_state"] == "Open"
    assert status["data_delay_minutes"] == 15
