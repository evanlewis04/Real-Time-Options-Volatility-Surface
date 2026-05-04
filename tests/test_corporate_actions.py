import pandas as pd

import src.quant.corporate_actions as corporate_actions_module
from src.quant.corporate_actions import (
    CorporateActionProvider,
    LocalCorporateActionSource,
    expiry_corporate_action_metadata,
)


def test_local_corporate_action_source_loads_dividends_and_splits(tmp_path):
    path = tmp_path / "actions.csv"
    path.write_text(
        "symbol,action_type,effective_date,description,value,ratio\n"
        "AAPL,dividend,2026-06-01,Cash dividend,0.26,\n"
        "AAPL,split,2026-07-01,Forward split,,2:1\n"
        "MSFT,dividend,2026-06-05,Cash dividend,0.83,\n",
        encoding="utf-8",
    )

    snapshot = LocalCorporateActionSource(path).load("aapl")

    assert snapshot.symbol == "AAPL"
    assert snapshot.mode == "Local"
    assert [event.action_type for event in snapshot.events] == ["dividend", "split"]
    assert snapshot.events[1].ratio == "2:1"
    assert len(snapshot.warning_messages()) == 2


def test_expiry_corporate_action_metadata_maps_events_to_expiry(tmp_path):
    path = tmp_path / "actions.csv"
    path.write_text(
        "symbol,action_type,effective_date,description,value,ratio\n"
        "AAPL,dividend,2026-06-01,Cash dividend,0.26,\n"
        "AAPL,split,2026-07-01,Forward split,,2:1\n",
        encoding="utf-8",
    )
    snapshot = LocalCorporateActionSource(path).load("AAPL")

    metadata = expiry_corporate_action_metadata(
        [pd.Timestamp("2026-06-19"), pd.Timestamp("2026-07-17")],
        snapshot,
    )

    assert [event["action_type"] for event in metadata["2026-06-19"]] == ["dividend"]
    assert [event["action_type"] for event in metadata["2026-07-17"]] == ["dividend", "split"]


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
