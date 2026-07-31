from datetime import datetime

import pandas as pd

from src.quant.events import EventCalendarProvider, expiry_event_metadata


def test_local_event_calendar_loads_symbol_and_macro_events(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "\n".join(
            [
                "symbol,event_type,event_date,description,source",
                "*,fomc,2026-06-17,FOMC decision,fixture",
                "AAPL,earnings,2026-07-30,AAPL earnings,fixture",
                "MSFT,earnings,2026-07-28,MSFT earnings,fixture",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = EventCalendarProvider(local_path=path).get("AAPL")
    events = [event.as_dict() for event in snapshot.events]

    assert [event["event_type"] for event in events] == ["fomc", "earnings"]
    assert events[0]["symbol"] == "*"
    assert events[1]["symbol"] == "AAPL"


def test_expiry_event_metadata_maps_events_through_each_expiry(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "\n".join(
            [
                "symbol,event_type,event_date,description,source",
                "*,cpi,2026-06-10,CPI release,fixture",
                "*,fomc,2026-06-17,FOMC decision,fixture",
                "AAPL,earnings,2026-07-30,AAPL earnings,fixture",
            ]
        ),
        encoding="utf-8",
    )
    snapshot = EventCalendarProvider(local_path=path, as_of=datetime(2026, 6, 1)).get("AAPL")
    expiries = pd.Series([datetime(2026, 6, 19), datetime(2026, 8, 1)])

    metadata = expiry_event_metadata(expiries, snapshot)

    assert [event["event_type"] for event in metadata["2026-06-19"]] == ["cpi", "fomc"]
    assert [event["event_type"] for event in metadata["2026-08-01"]] == ["cpi", "fomc", "earnings"]
