"""Local event-calendar awareness for option expiries."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd


DEFAULT_EVENT_CALENDAR_PATH = Path("config/events.csv")
EventType = Literal["earnings", "fomc", "cpi", "dividend", "corporate_action", "other"]


@dataclass(frozen=True)
class MarketEvent:
    """A symbol-specific or macro event that can affect option expiries."""

    symbol: str
    event_type: EventType
    event_date: date
    description: str
    source: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "event_type": self.event_type,
            "event_date": self.event_date.isoformat(),
            "description": self.description,
            "source": self.source,
        }


@dataclass(frozen=True)
class EventCalendarSnapshot:
    """Event calendar events and provenance for one symbol."""

    symbol: str
    as_of: datetime
    source: str
    mode: str
    events: tuple[MarketEvent, ...] = field(default_factory=tuple)
    fallback_reason: Optional[str] = None

    def upcoming(self, horizon_days: int = 370) -> tuple[MarketEvent, ...]:
        start = self.as_of.date()
        end = start + timedelta(days=horizon_days)
        return tuple(event for event in self.events if start <= event.event_date <= end)

    def through_expiry(self, expiry: Any, extra_events: tuple[MarketEvent, ...] = ()) -> tuple[MarketEvent, ...]:
        expiry_date = pd.to_datetime(expiry, errors="coerce")
        if pd.isna(expiry_date):
            return ()
        start = self.as_of.date()
        end = expiry_date.date()
        events = tuple(self.events) + tuple(extra_events)
        return tuple(event for event in events if start <= event.event_date <= end)

    def metadata_dict(self) -> dict[str, Any]:
        upcoming = self.upcoming()
        return {
            "event_source": self.source,
            "event_mode": self.mode,
            "event_timestamp": self.as_of,
            "event_fallback_reason": self.fallback_reason,
            "event_count": len(upcoming),
            "events": [event.as_dict() for event in upcoming],
        }


class LocalEventCalendarSource:
    """Load macro and symbol events from a local CSV file."""

    def __init__(self, path: Path | str = DEFAULT_EVENT_CALENDAR_PATH, as_of: datetime | None = None):
        self.path = Path(path)
        # When set, pins the snapshot's as-of reference instead of deriving it
        # from the wall clock / file mtime. Lets tests be deterministic.
        self.as_of = as_of

    def load(self, symbol: str) -> EventCalendarSnapshot:
        key = symbol.upper()
        if not self.path.exists():
            return EventCalendarSnapshot(
                symbol=key,
                as_of=self.as_of or datetime.now(),
                source=f"local:{self.path}",
                mode="Local",
                fallback_reason="Local event calendar file missing; no events loaded",
            )

        frame = pd.read_csv(self.path)
        if frame.empty:
            return EventCalendarSnapshot(
                symbol=key,
                as_of=self.as_of or datetime.fromtimestamp(self.path.stat().st_mtime),
                source=f"local:{self.path}",
                mode="Local",
                fallback_reason="Local event calendar is empty; no events loaded",
            )

        lower_cols = {str(col).strip().lower(): col for col in frame.columns}
        events = _events_from_rows(frame, lower_cols, key, f"local:{self.path}")
        return EventCalendarSnapshot(
            symbol=key,
            as_of=self.as_of or datetime.fromtimestamp(self.path.stat().st_mtime),
            source=f"local:{self.path}",
            mode="Local",
            events=tuple(events),
        )


class EventCalendarProvider:
    """Cached local event-calendar provider."""

    def __init__(
        self,
        local_path: Path | str = DEFAULT_EVENT_CALENDAR_PATH,
        cache_ttl_seconds: int = 3600,
        as_of: datetime | None = None,
    ):
        self.local_source = LocalEventCalendarSource(
            os.getenv("ROVS_EVENT_CALENDAR_PATH") or local_path,
            as_of=as_of,
        )
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache: dict[str, tuple[EventCalendarSnapshot, datetime]] = {}

    def get(self, symbol: str, force_refresh: bool = False) -> EventCalendarSnapshot:
        key = symbol.upper()
        now = datetime.now()
        cached = self._cache.get(key)
        if cached and not force_refresh and now - cached[1] < self.cache_ttl:
            return cached[0]
        snapshot = self.local_source.load(key)
        self._cache[key] = (snapshot, now)
        return snapshot

    def clear_cache(self) -> None:
        self._cache.clear()


def expiry_event_metadata(
    expiries: pd.Series | list[Any],
    snapshot: EventCalendarSnapshot,
    extra_events: tuple[MarketEvent, ...] = (),
) -> dict[str, list[dict[str, Any]]]:
    """Build expiry-date to event map for term-structure annotations."""
    expiry_values = pd.to_datetime(pd.Series(expiries), errors="coerce").dropna().dt.date.unique()
    out: dict[str, list[dict[str, Any]]] = {}
    for expiry in sorted(expiry_values):
        events = snapshot.through_expiry(expiry, extra_events)
        if events:
            out[expiry.isoformat()] = [event.as_dict() for event in events]
    return out


def _events_from_rows(
    frame: pd.DataFrame,
    lower_cols: dict[str, str],
    symbol: str,
    source: str,
) -> list[MarketEvent]:
    symbol_col = lower_cols.get("symbol")
    type_col = lower_cols.get("event_type") or lower_cols.get("type")
    date_col = lower_cols.get("event_date") or lower_cols.get("date")
    description_col = lower_cols.get("description")
    source_col = lower_cols.get("source")
    if symbol_col is None or type_col is None or date_col is None:
        return []

    rows = frame[
        frame[symbol_col].astype(str).str.upper().isin({symbol.upper(), "*", "GLOBAL", "MACRO"})
    ]
    events: list[MarketEvent] = []
    for _, row in rows.iterrows():
        event_date = pd.to_datetime(row.get(date_col), errors="coerce")
        if pd.isna(event_date):
            continue
        raw_type = str(row.get(type_col) or "other").strip().lower()
        event_type: EventType = raw_type if raw_type in _allowed_types() else "other"  # type: ignore[assignment]
        event_symbol = str(row.get(symbol_col) or "*").upper()
        description = (
            str(row.get(description_col))
            if description_col and pd.notna(row.get(description_col))
            else event_type.title()
        )
        event_source = str(row.get(source_col) or source) if source_col else source
        events.append(
            MarketEvent(
                symbol=event_symbol,
                event_type=event_type,
                event_date=event_date.date(),
                description=description,
                source=event_source,
            )
        )
    return sorted(events, key=lambda event: (event.event_date, event.event_type, event.symbol))


def _allowed_types() -> set[str]:
    return {"earnings", "fomc", "cpi", "dividend", "corporate_action", "other"}
