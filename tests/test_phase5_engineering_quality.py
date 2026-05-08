import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from dashboard_connector import DashboardConnector
from src.analysis.surface_builder import build_surface
from src.config.settings import load_app_settings
from src.data.demo_provider import DemoOptionsProvider
from src.data.models import MarketDataSnapshot
from src.data.options_provider import OptionsChainMetadata, YFinanceOptionsProvider
from src.data.price_provider import RealTimePriceProvider
from src.utils.structured_logging import StructuredJsonFormatter
from tests.fixtures.option_chain_fixture import FIXTURE_NOW, raw_yfinance_option_chain


class StaticPriceProvider(RealTimePriceProvider):
    def __init__(self):
        self.cache_duration = 60
        self.price_cache = {}
        self.cache_timestamps = {}
        self.price_movements = {}
        self.last_movement_update = FIXTURE_NOW
        self.current_market_prices = {"AAPL": 200.0}
        self.yfinance_working = False

    def get_live_price(self, symbol: str) -> float:
        return 200.0


def test_typed_settings_validate_defaults_and_environment_overrides():
    settings = load_app_settings(
        {
            "VOL_SURFACE_MAX_EXPIRATIONS": "3",
            "VOL_SURFACE_CHAIN_CACHE_SECONDS": "42",
            "VOL_SURFACE_DEMO_RANDOM_SEED": "99",
            "VOL_SURFACE_SNAPSHOT_DIR": "tmp/snapshots",
            "VOL_SURFACE_STRUCTURED_LOGS": "false",
        }
    )

    assert settings.providers.max_expirations == 3
    assert settings.providers.chain_cache_seconds == 42
    assert settings.demo.random_seed == 99
    assert settings.dashboard.snapshot_dir.parts[-2:] == ("tmp", "snapshots")
    assert settings.logging.structured is False

    with pytest.raises(ValueError):
        load_app_settings({"VOL_SURFACE_MAX_EXPIRATIONS": "0"})


def test_structured_log_formatter_preserves_provider_fields():
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord("test", logging.INFO, __file__, 10, "provider fetch", (), None)
    record.structured = {
        "event": "provider_fetch",
        "symbol": "AAPL",
        "provider": "fixture",
        "latency_ms": 12.5,
        "cache_hit": False,
        "fallback_reason": None,
    }

    payload = json.loads(formatter.format(record))

    assert payload["event"] == "provider_fetch"
    assert payload["symbol"] == "AAPL"
    assert payload["provider"] == "fixture"
    assert payload["latency_ms"] == 12.5
    assert payload["cache_hit"] is False


def test_demo_provider_is_named_deterministic_and_contract_compatible():
    price_provider = StaticPriceProvider()
    provider_a = DemoOptionsProvider(price_provider, random_seed=7, max_expirations=2)
    provider_b = DemoOptionsProvider(price_provider, random_seed=7, max_expirations=2)

    first, first_meta = provider_a.fetch_chain("aapl", 200.0, as_of=FIXTURE_NOW)
    second, second_meta = provider_b.fetch_chain("aapl", 200.0, as_of=FIXTURE_NOW)

    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert first_meta.source == "demo synthetic provider"
    assert first_meta.mode == "Synthetic"
    assert first_meta.valid_rows == len(first)
    snapshot = MarketDataSnapshot.from_chain_frame("AAPL", 200.0, FIXTURE_NOW, first, first_meta.as_dict())
    assert snapshot.source == "demo synthetic provider"
    assert len(snapshot.options) == len(first)
    assert snapshot.options[0].contract.startswith("AAPL")
    assert second_meta.as_dict()["source"] == "demo synthetic provider"


def test_yfinance_provider_contract_with_offline_fixture():
    raw = raw_yfinance_option_chain()
    clean = YFinanceOptionsProvider._normalize(raw, "AAPL", 200.0, FIXTURE_NOW)
    meta = OptionsChainMetadata(
        symbol="AAPL",
        source="fixture",
        mode="Fixture",
        timestamp=FIXTURE_NOW,
        raw_rows=len(raw),
        valid_rows=len(clean),
        rejected_rows=len(raw) - len(clean),
        data_quality_score=clean.attrs["data_quality_score"],
    )

    snapshot = MarketDataSnapshot.from_chain_frame("AAPL", 200.0, FIXTURE_NOW, clean, meta.as_dict())

    assert not clean.empty
    assert snapshot.symbol == "AAPL"
    assert snapshot.source == "fixture"
    assert snapshot.valid_rows == len(clean)
    assert snapshot.options_frame().iloc[0]["contractSymbol"]


def test_dashboard_connector_records_provider_and_surface_timings(tmp_path):
    connector = DashboardConnector()
    connector.price_provider = StaticPriceProvider()
    connector.demo_provider = DemoOptionsProvider(connector.price_provider, random_seed=11, max_expirations=2)
    connector.options_provider = connector.demo_provider
    connector.snapshot_dir = tmp_path
    connector.chain_cache.clear()

    frame, meta = connector.get_options_chain_snapshot("AAPL")
    connector.get_vol_surface_data("AAPL")
    health = connector.get_system_health()

    assert not frame.empty
    assert meta["source"] == "demo synthetic provider"
    assert health["performance"]["slowest_steps"]
    assert any(item["operation"] == "options_chain_fetch" for item in health["performance"]["recent_steps"])
    assert any(item["operation"] == "surface_build" for item in health["performance"]["recent_steps"])


def test_surface_builder_handles_phase5_fallback_cases():
    strikes, expiries, vols = build_surface(pd.DataFrame(), 200.0, "AAPL")
    assert strikes.size > 0
    assert expiries.size > 0
    assert np.isfinite(vols).all()

    missing = pd.DataFrame({"strike": [200.0], "impliedVolatility": [0.25]})
    _, _, missing_vols = build_surface(missing, 200.0, "AAPL")
    assert np.isfinite(missing_vols).all()

    fixture = YFinanceOptionsProvider._normalize(raw_yfinance_option_chain(), "AAPL", 200.0, FIXTURE_NOW)
    fixture.loc[fixture.index[:3], "impliedVolatility"] = np.nan
    fixture = fixture.dropna(subset=["impliedVolatility"]).head(4)
    sparse_strikes, sparse_expiries, sparse_vols = build_surface(fixture, 200.0, "AAPL")

    assert sparse_strikes.size > 0
    assert sparse_expiries.size > 0
    assert np.isfinite(sparse_vols).all()
