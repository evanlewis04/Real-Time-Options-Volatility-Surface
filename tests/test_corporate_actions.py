from datetime import timedelta

import pandas as pd

import src.quant.corporate_actions as corporate_actions_module
from src.quant.corporate_actions import (
    CorporateActionProvider,
    LocalCorporateActionSource,
    expiry_corporate_action_metadata,
)


def _future(days: int) -> pd.Timestamp:
    """A date relative to today so fixtures never expire into the past."""

    return pd.Timestamp.today().normalize() + timedelta(days=days)


def test_local_corporate_action_source_loads_dividends_and_splits(tmp_path):
    dividend = _future(10).strftime("%Y-%m-%d")
    split = _future(40).strftime("%Y-%m-%d")
    msft_dividend = _future(14).strftime("%Y-%m-%d")
    path = tmp_path / "actions.csv"
    path.write_text(
        "symbol,action_type,effective_date,description,value,ratio\n"
        f"AAPL,dividend,{dividend},Cash dividend,0.26,\n"
        f"AAPL,split,{split},Forward split,,2:1\n"
        f"MSFT,dividend,{msft_dividend},Cash dividend,0.83,\n",
        encoding="utf-8",
    )

    snapshot = LocalCorporateActionSource(path).load("aapl")

    assert snapshot.symbol == "AAPL"
    assert snapshot.mode == "Local"
    assert [event.action_type for event in snapshot.events] == ["dividend", "split"]
    assert snapshot.events[1].ratio == "2:1"
    assert len(snapshot.warning_messages()) == 2


def test_expiry_corporate_action_metadata_maps_events_to_expiry(tmp_path):
    dividend = _future(10)
    split = _future(40)
    expiry_one = _future(20)
    expiry_two = _future(50)
    path = tmp_path / "actions.csv"
    path.write_text(
        "symbol,action_type,effective_date,description,value,ratio\n"
        f"AAPL,dividend,{dividend.strftime('%Y-%m-%d')},Cash dividend,0.26,\n"
        f"AAPL,split,{split.strftime('%Y-%m-%d')},Forward split,,2:1\n",
        encoding="utf-8",
    )
    snapshot = LocalCorporateActionSource(path).load("AAPL")

    metadata = expiry_corporate_action_metadata([expiry_one, expiry_two], snapshot)

    assert [event["action_type"] for event in metadata[expiry_one.strftime("%Y-%m-%d")]] == ["dividend"]
    assert [event["action_type"] for event in metadata[expiry_two.strftime("%Y-%m-%d")]] == ["dividend", "split"]


def test_provider_falls_back_to_local_when_live_actions_are_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "actions.csv"
    path.write_text("symbol,action_type,effective_date,description\nAAPL,split,2026-07-01,Forward split\n")
    monkeypatch.setattr(
        corporate_actions_module.YFinanceCorporateActionSource,
        "load",
        lambda self, symbol: (_ for _ in ()).throw(RuntimeError("network unavailable")),
    )

    snapshot = CorporateActionProvider(preferred_source="yfinance", local_path=path).get("AAPL")

    assert snapshot.mode == "Fallback"
    assert "yfinance" in snapshot.fallback_reason
    assert snapshot.events[0].action_type == "split"
