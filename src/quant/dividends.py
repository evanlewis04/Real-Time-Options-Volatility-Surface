"""Dividend assumptions for equity option pricing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


DEFAULT_DIVIDEND_PATH = Path("config/dividends.csv")


@dataclass(frozen=True)
class DividendEvent:
    """A known or projected discrete cash dividend."""

    ex_date: date
    amount: float
    currency: str = "USD"


@dataclass(frozen=True)
class DividendAssumption:
    """Symbol-level dividend inputs and provenance."""

    symbol: str
    annual_yield: float
    as_of: datetime
    source: str
    mode: str
    events: tuple[DividendEvent, ...] = field(default_factory=tuple)
    fallback_reason: Optional[str] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def events_until(self, expiry: Any) -> tuple[DividendEvent, ...]:
        """Return discrete dividends with ex-dates through the option expiry."""
        expiry_date = pd.to_datetime(expiry, errors="coerce")
        if pd.isna(expiry_date):
            return ()
        start = self.as_of.date()
        end = expiry_date.date()
        return tuple(event for event in self.events if start <= event.ex_date <= end)

    def discrete_amount_until(self, expiry: Any) -> float:
        """Return undiscounted cash dividends through an expiry."""
        return float(sum(event.amount for event in self.events_until(expiry)))

    def present_value_until(self, expiry: Any, risk_free_rate: float | None = None) -> float:
        """Return present value of discrete dividends through an expiry."""
        rate = float(risk_free_rate or 0.0)
        total = 0.0
        for event in self.events_until(expiry):
            days = max(0, (event.ex_date - self.as_of.date()).days)
            total += event.amount * float(np.exp(-rate * days / 365.0))
        return float(total)

    def effective_yield(self, expiry: Any, spot: float, risk_free_rate: float | None = None) -> float:
        """Blend continuous yield with discrete dividends as BSM-compatible yield."""
        expiry_dt = pd.to_datetime(expiry, errors="coerce")
        if pd.isna(expiry_dt) or spot <= 0:
            return self.annual_yield

        dte = max(0, (expiry_dt.date() - self.as_of.date()).days)
        if dte <= 0:
            return self.annual_yield

        pv = min(self.present_value_until(expiry_dt, risk_free_rate), spot * 0.95)
        if pv <= 0:
            return self.annual_yield

        discrete_yield = -np.log(max((spot - pv) / spot, 1e-9)) / (dte / 365.0)
        return float(max(0.0, self.annual_yield + discrete_yield))

    def metadata_dict(self) -> dict[str, Any]:
        """Return dashboard-friendly dividend provenance."""
        return {
            "dividend_source": self.source,
            "dividend_mode": self.mode,
            "dividend_timestamp": self.as_of,
            "dividend_fallback_reason": self.fallback_reason,
            "annual_dividend_yield": self.annual_yield,
            "dividend_events": [
                {"ex_date": event.ex_date.isoformat(), "amount": event.amount, "currency": event.currency}
                for event in self.events
            ],
            "dividend_warnings": list(self.warnings),
        }


class DividendSourceError(RuntimeError):
    """Raised when a dividend source cannot produce assumptions."""


class LocalDividendSource:
    """Load dividend assumptions from a local CSV file."""

    def __init__(self, path: Path | str = DEFAULT_DIVIDEND_PATH):
        self.path = Path(path)

    def load(self, symbol: str) -> DividendAssumption:
        key = symbol.upper()
        if not self.path.exists():
            return _zero_dividend(
                key,
                source=f"local:{self.path}",
                fallback_reason="Local dividend file missing; assuming no dividends",
            )

        frame = pd.read_csv(self.path)
        if frame.empty:
            return _zero_dividend(
                key,
                source=f"local:{self.path}",
                fallback_reason="Local dividend file is empty; assuming no dividends",
            )

        lower_cols = {str(col).strip().lower(): col for col in frame.columns}
        symbol_col = lower_cols.get("symbol")
        if symbol_col is None:
            return _zero_dividend(
                key,
                source=f"local:{self.path}",
                fallback_reason="Local dividend file has no symbol column; assuming no dividends",
            )

        rows = frame[frame[symbol_col].astype(str).str.upper() == key].copy()
        if rows.empty:
            return _zero_dividend(key, source=f"local:{self.path}")

        annual_yield = _first_numeric(rows, lower_cols, ("annual_yield", "dividend_yield", "yield"), default=0.0)
        events = _events_from_rows(rows, lower_cols)
        return DividendAssumption(
            symbol=key,
            annual_yield=max(0.0, annual_yield),
            as_of=datetime.fromtimestamp(self.path.stat().st_mtime),
            source=f"local:{self.path}",
            mode="Local",
            events=events,
        )


class YFinanceDividendSource:
    """Optional live dividend assumptions from yfinance."""

    def load(self, symbol: str) -> DividendAssumption:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise DividendSourceError("yfinance is not installed") from exc

        key = symbol.upper()
        ticker = yf.Ticker(key)
        info = getattr(ticker, "info", {}) or {}
        annual_yield = _parse_rate(info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0.0)
        events = _events_from_yfinance(getattr(ticker, "dividends", pd.Series(dtype=float)))
        return DividendAssumption(
            symbol=key,
            annual_yield=max(0.0, annual_yield),
            as_of=datetime.now(),
            source="yfinance dividends",
            mode="Live/Delayed",
            events=events,
        )


class DividendProvider:
    """Cached dividend provider with offline-safe local assumptions."""

    def __init__(
        self,
        preferred_source: str | None = None,
        local_path: Path | str = DEFAULT_DIVIDEND_PATH,
        cache_ttl_seconds: int = 3600,
    ):
        self.preferred_source = (preferred_source or os.getenv("ROVS_DIVIDEND_SOURCE") or "local").lower()
        self.local_source = LocalDividendSource(local_path)
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cache: dict[str, tuple[DividendAssumption, datetime]] = {}

    def get(self, symbol: str, force_refresh: bool = False) -> DividendAssumption:
        """Return dividend assumptions for a symbol."""
        key = symbol.upper()
        now = datetime.now()
        cached = self._cache.get(key)
        if cached and not force_refresh and now - cached[1] < self.cache_ttl:
            return cached[0]

        assumption = self._load(key)
        self._cache[key] = (assumption, now)
        return assumption

    def clear_cache(self) -> None:
        """Clear cached dividend assumptions."""
        self._cache.clear()

    def _load(self, symbol: str) -> DividendAssumption:
        if self.preferred_source in {"yfinance", "live"}:
            try:
                return YFinanceDividendSource().load(symbol)
            except Exception as exc:
                local = self.local_source.load(symbol)
                return DividendAssumption(
                    symbol=local.symbol,
                    annual_yield=local.annual_yield,
                    as_of=local.as_of,
                    source=local.source,
                    mode="Fallback",
                    events=local.events,
                    fallback_reason=f"Live yfinance dividends unavailable: {exc}",
                    warnings=local.warnings,
                )
        return self.local_source.load(symbol)


def apply_dividends_to_options(
    frame: pd.DataFrame,
    assumption: DividendAssumption,
    spot: float,
) -> pd.DataFrame:
    """Attach dividend yield and discrete dividend fields to option rows."""
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    expiries = pd.to_datetime(enriched.get("expiration"), errors="coerce")
    risk_free = pd.to_numeric(enriched.get("riskFreeRate"), errors="coerce")
    if risk_free.empty:
        risk_free = pd.Series([0.0] * len(enriched), index=enriched.index)

    effective = []
    amounts = []
    pvs = []
    counts = []
    for expiry, rate in zip(expiries, risk_free):
        rate_value = float(rate) if pd.notna(rate) else 0.0
        events = assumption.events_until(expiry)
        amount = assumption.discrete_amount_until(expiry)
        pv = assumption.present_value_until(expiry, rate_value)
        effective.append(assumption.effective_yield(expiry, spot, rate_value))
        amounts.append(amount)
        pvs.append(pv)
        counts.append(len(events))

    enriched["dividendYield"] = assumption.annual_yield
    enriched["effectiveDividendYield"] = effective
    enriched["discreteDividendAmount"] = amounts
    enriched["discreteDividendPV"] = pvs
    enriched["discreteDividendCount"] = counts
    return enriched


def expiry_dividend_metadata(
    frame: pd.DataFrame,
    assumption: DividendAssumption,
    spot: float,
) -> dict[str, dict[str, float | int]]:
    """Build expiry-level dividend metadata from an options chain."""
    if frame.empty or "expiration" not in frame.columns:
        return {}

    work = apply_dividends_to_options(frame, assumption, spot)
    expiries = pd.to_datetime(work["expiration"], errors="coerce")
    out: dict[str, dict[str, float | int]] = {}
    for expiry in sorted(expiries.dropna().dt.date.unique()):
        sub = work[expiries.dt.date == expiry]
        out[expiry.isoformat()] = {
            "annual_yield": float(sub["dividendYield"].median()),
            "effective_yield": float(sub["effectiveDividendYield"].median()),
            "discrete_amount": float(sub["discreteDividendAmount"].median()),
            "discrete_present_value": float(sub["discreteDividendPV"].median()),
            "discrete_count": int(sub["discreteDividendCount"].median()),
        }
    return out


def _zero_dividend(symbol: str, source: str, fallback_reason: str | None = None) -> DividendAssumption:
    return DividendAssumption(
        symbol=symbol.upper(),
        annual_yield=0.0,
        as_of=datetime.now(),
        source=source,
        mode="Local",
        fallback_reason=fallback_reason,
    )


def _first_numeric(
    frame: pd.DataFrame,
    lower_cols: dict[str, str],
    names: tuple[str, ...],
    default: float,
) -> float:
    for name in names:
        col = lower_cols.get(name)
        if col is None:
            continue
        values = pd.to_numeric(frame[col], errors="coerce").dropna()
        if not values.empty:
            return _parse_rate(values.iloc[0])
    return default


def _events_from_rows(frame: pd.DataFrame, lower_cols: dict[str, str]) -> tuple[DividendEvent, ...]:
    ex_col = lower_cols.get("ex_date") or lower_cols.get("date")
    amount_col = lower_cols.get("amount") or lower_cols.get("dividend")
    currency_col = lower_cols.get("currency")
    if ex_col is None or amount_col is None:
        return ()

    events: list[DividendEvent] = []
    for _, row in frame.iterrows():
        ex_date = pd.to_datetime(row.get(ex_col), errors="coerce")
        amount = pd.to_numeric(pd.Series([row.get(amount_col)]), errors="coerce").iloc[0]
        if pd.notna(ex_date) and pd.notna(amount) and float(amount) > 0:
            currency = str(row.get(currency_col) or "USD") if currency_col else "USD"
            events.append(DividendEvent(ex_date=ex_date.date(), amount=float(amount), currency=currency))
    return tuple(sorted(events, key=lambda event: event.ex_date))


def _events_from_yfinance(series: pd.Series) -> tuple[DividendEvent, ...]:
    if series is None or len(series) == 0:
        return ()
    events: list[DividendEvent] = []
    now = datetime.now().date()
    horizon = now + timedelta(days=370)
    for idx, amount in series.items():
        ex_date = pd.to_datetime(idx, errors="coerce")
        if pd.notna(ex_date) and now <= ex_date.date() <= horizon and float(amount) > 0:
            events.append(DividendEvent(ex_date=ex_date.date(), amount=float(amount)))
    return tuple(sorted(events, key=lambda event: event.ex_date))


def _parse_rate(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    rate = float(str(value).strip().replace("%", ""))
    return rate / 100.0 if abs(rate) > 1.0 else rate

