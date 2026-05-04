from datetime import datetime

from src.data.market_calendar import EASTERN, MarketCalendar, us_equity_holidays


def test_market_calendar_reports_open_regular_session():
    calendar = MarketCalendar()

    status = calendar.status(datetime(2026, 5, 4, 10, 0, tzinfo=EASTERN))

    assert status.is_open
    assert status.session_state == "Open"
    assert status.open_time.hour == 9
    assert status.close_time.hour == 16
    assert status.data_delay_minutes == 15


def test_market_calendar_reports_weekend_closed_and_next_open():
    calendar = MarketCalendar()

    status = calendar.status(datetime(2026, 5, 2, 12, 0, tzinfo=EASTERN))

    assert not status.is_open
    assert status.session_state == "Closed"
    assert status.reason == "weekend"
    assert status.next_open.date().isoformat() == "2026-05-04"


def test_market_calendar_reports_full_holiday_closed():
    calendar = MarketCalendar()

    status = calendar.status(datetime(2026, 12, 25, 12, 0, tzinfo=EASTERN))

    assert not status.is_open
    assert status.session_state == "Closed"
    assert status.reason == "holiday"
    assert status.next_open.date().isoformat() == "2026-12-28"


def test_market_calendar_reports_after_hours_previous_close():
    calendar = MarketCalendar()

    status = calendar.status(datetime(2026, 5, 4, 17, 0, tzinfo=EASTERN))

    assert not status.is_open
    assert status.session_state == "After-hours"
    assert status.previous_close.date().isoformat() == "2026-05-04"
    assert status.next_open.date().isoformat() == "2026-05-05"


def test_us_equity_holidays_include_good_friday_and_thanksgiving():
    holidays = us_equity_holidays(2026)

    assert "2026-04-03" in {day.isoformat() for day in holidays}
    assert "2026-11-26" in {day.isoformat() for day in holidays}
