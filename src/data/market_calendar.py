"""Market calendar support for US equity-style dashboard data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class MarketSessionStatus:
    """Current market-session state for diagnostics and data labels."""

    market: str
    timestamp: datetime
    is_open: bool
    session_state: str
    open_time: datetime | None
    close_time: datetime | None
    previous_close: datetime | None
    next_open: datetime | None
    reason: str
    data_delay_minutes: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "market": self.market,
            "timestamp": self.timestamp,
            "is_open": self.is_open,
            "session_state": self.session_state,
            "open_time": self.open_time,
            "close_time": self.close_time,
            "previous_close": self.previous_close,
            "next_open": self.next_open,
            "reason": self.reason,
            "data_delay_minutes": self.data_delay_minutes,
        }


class MarketCalendar:
    """NYSE-compatible calendar with optional pandas_market_calendars backend."""

    def __init__(self, market: str = "XNYS", data_delay_minutes: int = 15):
        self.market = market
        self.data_delay_minutes = data_delay_minutes
        self._calendar = self._load_calendar(market)

    def status(self, at: datetime | None = None) -> MarketSessionStatus:
        """Return current regular-session status."""
        now = self._to_eastern(at or datetime.now(tz=EASTERN))
        if self._calendar is not None:
            status = self._status_from_pandas_market_calendar(now)
            if status is not None:
                return status
        return self._fallback_status(now)

    def _status_from_pandas_market_calendar(self, now: datetime) -> MarketSessionStatus | None:
        try:
            start = (now.date() - timedelta(days=7)).isoformat()
            end = (now.date() + timedelta(days=7)).isoformat()
            schedule = self._calendar.schedule(start_date=start, end_date=end)
            today_rows = schedule[schedule.index.date == now.date()]
            today_open = None
            today_close = None
            reason = "regular session"
            if not today_rows.empty:
                today_open = today_rows.iloc[0]["market_open"].to_pydatetime().astimezone(EASTERN)
                today_close = today_rows.iloc[0]["market_close"].to_pydatetime().astimezone(EASTERN)
                if today_open <= now <= today_close:
                    state = "Open"
                    is_open = True
                elif now < today_open:
                    state = "Pre-market"
                    is_open = False
                else:
                    state = "After-hours"
                    is_open = False
            else:
                state = "Closed"
                is_open = False
                reason = self._closed_reason(now.date())

            previous_close = None
            next_open = None
            previous_rows = schedule[schedule["market_close"] < now.astimezone(UTC)]
            next_rows = schedule[schedule["market_open"] > now.astimezone(UTC)]
            if not previous_rows.empty:
                previous_close = previous_rows.iloc[-1]["market_close"].to_pydatetime().astimezone(EASTERN)
            if not next_rows.empty:
                next_open = next_rows.iloc[0]["market_open"].to_pydatetime().astimezone(EASTERN)

            return MarketSessionStatus(
                market=self.market,
                timestamp=now,
                is_open=is_open,
                session_state=state,
                open_time=today_open,
                close_time=today_close,
                previous_close=previous_close,
                next_open=next_open,
                reason=reason,
                data_delay_minutes=self.data_delay_minutes,
            )
        except Exception:
            return None

    def _fallback_status(self, now: datetime) -> MarketSessionStatus:
        session_day = now.date()
        open_time = self._session_open(session_day) if self._is_session_day(session_day) else None
        close_time = self._session_close(session_day) if self._is_session_day(session_day) else None

        if open_time and close_time and open_time <= now <= close_time:
            state = "Open"
            is_open = True
            reason = "regular session"
        elif open_time and now < open_time:
            state = "Pre-market"
            is_open = False
            reason = "before regular session"
        elif close_time and now > close_time:
            state = "After-hours"
            is_open = False
            reason = "after regular session"
        else:
            state = "Closed"
            is_open = False
            reason = self._closed_reason(session_day)

        return MarketSessionStatus(
            market=self.market,
            timestamp=now,
            is_open=is_open,
            session_state=state,
            open_time=open_time,
            close_time=close_time,
            previous_close=self._previous_close(now),
            next_open=self._next_open(now),
            reason=reason,
            data_delay_minutes=self.data_delay_minutes,
        )

    def _is_session_day(self, value: date) -> bool:
        return value.weekday() < 5 and value not in us_equity_holidays(value.year)

    @staticmethod
    def _closed_reason(value: date) -> str:
        return "holiday" if value in us_equity_holidays(value.year) else "weekend"

    @staticmethod
    def _session_open(value: date) -> datetime:
        return datetime.combine(value, time(9, 30), tzinfo=EASTERN)

    @staticmethod
    def _session_close(value: date) -> datetime:
        return datetime.combine(value, time(16, 0), tzinfo=EASTERN)

    def _previous_close(self, now: datetime) -> datetime | None:
        current = now.date()
        if self._is_session_day(current) and now > self._session_close(current):
            return self._session_close(current)
        for offset in range(1, 15):
            candidate = current - timedelta(days=offset)
            if self._is_session_day(candidate):
                return self._session_close(candidate)
        return None

    def _next_open(self, now: datetime) -> datetime | None:
        current = now.date()
        if self._is_session_day(current) and now < self._session_open(current):
            return self._session_open(current)
        for offset in range(1, 15):
            candidate = current + timedelta(days=offset)
            if self._is_session_day(candidate):
                return self._session_open(candidate)
        return None

    @staticmethod
    def _to_eastern(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=EASTERN)
        return value.astimezone(EASTERN)

    @staticmethod
    def _load_calendar(market: str):
        try:
            import pandas_market_calendars as mcal

            return mcal.get_calendar(market)
        except Exception:
            return None


def us_equity_holidays(year: int) -> set[date]:
    """Return common full-day US equity market holidays for ``year``."""
    holidays = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _good_friday(year),
        _last_weekday(year, 5, 0),
        _observed(date(year, 6, 19)),
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    return {item for item in holidays if item.year == year}


def _observed(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    current = date(year, month, 1)
    while current.weekday() != weekday:
        current += timedelta(days=1)
    return current + timedelta(days=7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while current.weekday() != weekday:
        current -= timedelta(days=1)
    return current


def _good_friday(year: int) -> date:
    # Anonymous Gregorian algorithm for Easter Sunday, then subtract two days.
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day) - timedelta(days=2)
