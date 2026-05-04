from datetime import datetime

import pandas as pd
import pytest

from src.quant.dividends import (
    DividendAssumption,
    DividendEvent,
    DividendProvider,
    LocalDividendSource,
    apply_dividends_to_options,
    expiry_dividend_metadata,
)
import src.quant.dividends as dividends_module


def test_local_dividend_source_loads_symbol_yield_and_events(tmp_path):
    path = tmp_path / "dividends.csv"
    path.write_text(
        "symbol,annual_yield,ex_date,amount,currency\n"
        "AAPL,0.005,2026-05-15,0.26,USD\n"
        "MSFT,0.007,2026-05-20,0.83,USD\n",
        encoding="utf-8",
    )

    assumption = LocalDividendSource(path).load("aapl")

    assert assumption.symbol == "AAPL"
    assert assumption.mode == "Local"
    assert assumption.annual_yield == pytest.approx(0.005)
    assert len(assumption.events) == 1
    assert assumption.events[0].amount == pytest.approx(0.26)


def test_apply_dividends_to_options_adds_expiry_specific_effective_yields():
    assumption = DividendAssumption(
        symbol="AAPL",
        annual_yield=0.005,
        as_of=datetime(2026, 5, 3),
        source="fixture",
        mode="Local",
        events=(DividendEvent(pd.Timestamp("2026-05-15").date(), 0.26),),
    )
    frame = pd.DataFrame(
        [
            {"expiration": pd.Timestamp("2026-05-10"), "daysToExpiration": 7, "riskFreeRate": 0.05},
            {"expiration": pd.Timestamp("2026-06-19"), "daysToExpiration": 47, "riskFreeRate": 0.05},
        ]
    )

    enriched = apply_dividends_to_options(frame, assumption, spot=200.0)

    assert enriched.iloc[0]["discreteDividendAmount"] == 0.0
    assert enriched.iloc[1]["discreteDividendAmount"] == pytest.approx(0.26)
    assert enriched.iloc[1]["effectiveDividendYield"] > enriched.iloc[1]["dividendYield"]
    assert enriched.iloc[1]["discreteDividendCount"] == 1


def test_expiry_dividend_metadata_is_keyed_by_expiry_date():
    assumption = DividendAssumption(
        symbol="SPY",
        annual_yield=0.012,
        as_of=datetime(2026, 5, 3),
        source="fixture",
        mode="Local",
        events=(DividendEvent(pd.Timestamp("2026-06-20").date(), 1.85),),
    )
    frame = pd.DataFrame(
        [{"expiration": pd.Timestamp("2026-07-17"), "daysToExpiration": 75, "riskFreeRate": 0.04}]
    )

    metadata = expiry_dividend_metadata(frame, assumption, spot=500.0)

    assert metadata["2026-07-17"]["discrete_count"] == 1
    assert metadata["2026-07-17"]["discrete_amount"] == pytest.approx(1.85)


def test_provider_falls_back_to_local_when_live_source_is_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "dividends.csv"
    path.write_text("symbol,annual_yield\nAAPL,0.005\n", encoding="utf-8")
    monkeypatch.setattr(
        dividends_module.YFinanceDividendSource,
        "load",
        lambda self, symbol: (_ for _ in ()).throw(RuntimeError("network unavailable")),
    )

    provider = DividendProvider(preferred_source="yfinance", local_path=path)
    assumption = provider.get("AAPL")

    assert assumption.mode == "Fallback"
    assert "yfinance" in assumption.fallback_reason
    assert assumption.annual_yield == pytest.approx(0.005)
