"""Corporate-action awareness for market-data snapshots."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd


DEFAULT_CORPORATE_ACTION_PATH = Path("config/corporate_actions.csv")
ActionType = Literal["dividend", "split", "spinoff", "merger", "other"]


@dataclass(frozen=True)
class CorporateActionEvent:
    """A dividend, split, or other symbol-level corporate action."""

    symbol: str
    action_type: ActionType
    effective_date: date
    description: str
    value: Optional[float] = None
    ratio: Optional[str] = None
    source: str = "unknown"

    def warning(self) -> str:
        """Return a compact user-facing warning string."""
        detail = self.ratio if self.ratio else self.value
        suffix = f" ({detail})" if detail not in {None, ""} else ""
        return f"{self.symbol} {self.action_type} on {self.effective_date.isoformat()}: {self.description}{suffix}"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable event payload."""
        return {
            "symbol": self.symbol,
            "action_type": self.action_type,
            "effective_date": self.effective_date.isoformat(),
            "description": self.description,
            "value": self.value,
            "ratio": self.ratio,
            "source": self.source,
        }


@dataclass(frozen=True)
class CorporateActionSnapshot:
    """Corporate-action events and provenance for one symbol."""

    symbol: str
    as_of: datetime
    source: str
    mode: str
    events: tuple[CorporateActionEvent, ...] = field(default_factory=tuple)
    fallback_reason: Optional[str] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def upcoming(self, horizon_days: int = 370) -> tuple[CorporateActionEvent, ...]:
        """Return events from as_of through a horizon."""
        start = self.as_of.date()
        end = start + timedelta(days=horizon_days)
        return tuple(event for event in self.events if start <= event.effective_date <= end)

    def through_expiry(self, expiry: Any) -> tuple[CorporateActionEvent, ...]:
        """Return events effective on or before a given option expiry."""
        expiry_date = pd.to_datetime(expiry, errors="coerce")
        if pd.isna(expiry_date):
            return ()
        start = self.as_of.date()
        end = expiry_date.date()
        return tuple(event for event in self.events if start <= event.effective_date <= end)

    def warning_messages(self, horizon_days: int = 370) -> tuple[str, ...]:
        """Return warnings for upcoming actions."""
        event_warnings = tuple(event.warning() for event in self.upcoming(horizon_days))
        return tuple(self.warnings) + event_warnings

    def metadata_dict(self, horizon_days: int = 370) -> dict[str, Any]:
        """Return dashboard-friendly corporate-action metadata."""
        upcoming = self.upcoming(horizon_days)
        return {
            "corporate_action_source": self.source,
            "corporate_action_mode": self.mode,
            "corporate_action_timestamp": self.as_of,
            "corporate_action_fallback_reason": self.fallback_reason,
            "corporate_actions": [event.as_dict() for event in self.events],
            "upcoming_corporate_actions": [event.as_dict() for event in upcoming],
            "corporate_action_warning_count": len(upcoming) + len(self.warnings),
            "corporate_action_warnings": list(self.warning_messages(horizon_days)),
        }


class CorporateActionSourceError(RuntimeError):
    """Raised when a corporate-action source cannot produce data."""


class LocalCorporateActionSource:
    """Load corporate actions from a local CSV file."""

    def __init__(self, path: Path | str = DEFAULT_CORPORATE_ACTION_PATH):
        self.path = Path(path)

    def load(self, symbol: str) -> CorporateActionSnapshot:
        key = symbol.upper()
        if not self.path.exists():
            return _empty_snapshot(
                key,
                source=f"local:{self.path}",
                fallback_reason="Local corporate action file missing; no actions loaded",
            )

        frame = pd.read_csv(self.path)
        if frame.empty:
            return _empty_snapshot(
                key,
                source=f"local:{self.path}",
                fallback_reason="Local corporate action file is empty; no actions loaded",
            )

        lower_cols = {str(col).strip().lower(): col for col in frame.columns}
        symbol_col = lower_cols.get("symbol")
        if symbol_col is None:
            return _empty_snapshot(
                key,
                source=f"local:{self.path}",
                fallback_reason="Local corporate action file has no symbol column; no actions loaded",
            )

        rows = frame[frame[symbol_col].astype(str).str.upper() == key]
        events = tuple(_events_from_rows(rows, lower_cols, key, f"local:{self.path}"))
        return CorporateActionSnapshot(
            symbol=key,
            as_of=datetime.fromtimestamp(self.path.stat().st_mtime),
            source=f"local:{self.path}",
            mode="Local",
            events=events,
        )


class YFinanceCorporateActionSource:
    """Optional yfinance source for historical corporate actions."""

    def load(self, symbol: str) -> CorporateActionSnapshot:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise CorporateActionSourceError("yfinance is not installed") from exc

        key = symbol.upper()
        ticker = yf.Ticker(key)
        actions = getattr(ticker, "actions", pd.DataFrame())
        if actions is None or actions.empty:
            events: tuple[CorporateActionEvent, ...] = ()
        else:
            events = tuple(_events_from_yfinance(actions, key))

        return CorporateActionSnapshot(
            symbol=key,
            as_of=datetime.now(),
            source="yfinance actions",
            mode="Live/Delayed",
            events=events,
        )


class CorporateActionProvider:
    """Cached corporate-action provider with offline-safe local fallback."""

    def __init__(
        self,
        preferred_source: str | None = None,
        local_path: Path | str = DEFAULT_CORPORATE_ACTION_PATH,
        cache_ttl_seconds: int = 3600,
    ):
        self.preferred_source = (preferred_source or os.getenv("ROVS_CORPORATE_ACTION_SOURCE") or "local").lower()
        self.local_source = LocalCorporateActionSource(local_path)
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache: dict[str, tuple[CorporateActionSnapshot, datetime]] = {}

    def get(self, symbol: str, force_refresh: bool = False) -> CorporateActionSnapshot:
        """Return corporate actions for a symbol."""
        key = symbol.upper()
        now = datetime.now()
        cached = self._cache.get(key)
        if cached and not force_refresh and now - cached[1] < self.cache_ttl:
            return cached[0]

        snapshot = self._load(key)
        self._cache[key] = (snapshot, now)
        return snapshot

    def clear_cache(self) -> None:
        """Clear cached corporate actions."""
        self._cache.clear()

    def _load(self, symbol: str) -> CorporateActionSnapshot:
        if self.preferred_source in {"yfinance", "live"}:
            try:
                return YFinanceCorporateActionSource().load(symbol)
            except Exception as exc:
                local = self.local_source.load(symbol)
                return CorporateActionSnapshot(
                    symbol=local.symbol,
                    as_of=local.as_of,
                    source=local.source,
                    mode="Fallback",
                    events=local.events,
                    fallback_reason=f"Live yfinance corporate actions unavailable: {exc}",
                    warnings=local.warnings,
                )
        return self.local_source.load(symbol)


def expiry_corporate_action_metadata(
    expiries: pd.Series | list[Any],
    snapshot: CorporateActionSnapshot,
) -> dict[str, list[dict[str, Any]]]:
    """Build expiry-date to corporate-action event map."""
    expiry_values = pd.to_datetime(pd.Series(expiries), errors="coerce").dropna().dt.date.unique()
    out: dict[str, list[dict[str, Any]]] = {}
    for expiry in sorted(expiry_values):
        events = snapshot.through_expiry(expiry)
        if events:
            out[expiry.isoformat()] = [event.as_dict() for event in events]
    return out


def _empty_snapshot(
    symbol: str,
    source: str,
    fallback_reason: str | None = None,
) -> CorporateActionSnapshot:
    return CorporateActionSnapshot(
        symbol=symbol.upper(),
        as_of=datetime.now(),
        source=source,
        mode="Local",
        fallback_reason=fallback_reason,
    )


def _events_from_rows(
    rows: pd.DataFrame,
    lower_cols: dict[str, str],
    symbol: str,
    source: str,
) -> list[CorporateActionEvent]:
    type_col = lower_cols.get("action_type") or lower_cols.get("type")
    date_col = lower_cols.get("effective_date") or lower_cols.get("ex_date") or lower_cols.get("date")
    description_col = lower_cols.get("description")
    value_col = lower_cols.get("value") or lower_cols.get("amount")
    ratio_col = lower_cols.get("ratio")
    if type_col is None or date_col is None:
        return []

    events: list[CorporateActionEvent] = []
    for _, row in rows.iterrows():
        effective_date = pd.to_datetime(row.get(date_col), errors="coerce")
        if pd.isna(effective_date):
            continue

        raw_type = str(row.get(type_col) or "other").strip().lower()
        action_type: ActionType = raw_type if raw_type in _allowed_types() else "other"  # type: ignore[assignment]
        value = _float_or_none(row.get(value_col)) if value_col else None
        ratio = str(row.get(ratio_col)) if ratio_col and pd.notna(row.get(ratio_col)) else None
        description = (
            str(row.get(description_col))
            if description_col and pd.notna(row.get(description_col))
            else action_type.title()
        )
        events.append(
            CorporateActionEvent(
                symbol=symbol,
                action_type=action_type,
                effective_date=effective_date.date(),
                description=description,
                value=value,
                ratio=ratio,
                source=source,
            )
        )
    return sorted(events, key=lambda event: event.effective_date)


def _events_from_yfinance(actions: pd.DataFrame, symbol: str) -> list[CorporateActionEvent]:
    events: list[CorporateActionEvent] = []
    for idx, row in actions.iterrows():
        effective_date = pd.to_datetime(idx, errors="coerce")
        if pd.isna(effective_date):
            continue

        dividend = _float_or_none(row.get("Dividends"))
        if dividend and dividend > 0:
            events.append(
                CorporateActionEvent(
                    symbol=symbol,
                    action_type="dividend",
                    effective_date=effective_date.date(),
                    description="Cash dividend",
                    value=dividend,
                    source="yfinance actions",
                )
            )

        split = _float_or_none(row.get("Stock Splits"))
        if split and split > 0:
            events.append(
                CorporateActionEvent(
                    symbol=symbol,
                    action_type="split",
                    effective_date=effective_date.date(),
                    description="Stock split",
                    ratio=f"{split}:1",
                    source="yfinance actions",
                )
            )
    return sorted(events, key=lambda event: event.effective_date)


def _allowed_types() -> set[str]:
    return {"dividend", "split", "spinoff", "merger", "other"}


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)

