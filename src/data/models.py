"""Canonical market-data models for dashboard and provider boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Literal, Optional

import numpy as np
import pandas as pd


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionQuote:
    """Normalized option quote used by calculations and UI tables."""

    contract: str
    type: OptionType
    strike: float
    expiry: datetime
    dte: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    raw_iv: Optional[float] = None
    computed_iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    quote_timestamp: Optional[datetime] = None
    moneyness: Optional[float] = None
    bid_ask_spread: Optional[float] = None
    bid_ask_spread_pct: Optional[float] = None

    @classmethod
    def from_series(cls, row: pd.Series) -> "OptionQuote":
        expiry = pd.to_datetime(row.get("expiration"), errors="coerce")
        quote_ts = pd.to_datetime(row.get("quoteTimestamp"), errors="coerce")
        option_type = str(row.get("type", "")).lower()
        if option_type not in {"call", "put"}:
            raise ValueError(f"unsupported option type: {option_type!r}")

        return cls(
            contract=str(row.get("contractSymbol") or row.get("contract") or ""),
            type=option_type,  # type: ignore[arg-type]
            strike=_float_or_none(row.get("strike")) or 0.0,
            expiry=expiry.to_pydatetime() if pd.notna(expiry) else datetime.min,
            dte=int(_float_or_none(row.get("daysToExpiration")) or 0),
            bid=_float_or_none(row.get("bid")),
            ask=_float_or_none(row.get("ask")),
            mid=_float_or_none(row.get("mid")),
            last=_float_or_none(row.get("last")),
            volume=_int_or_none(row.get("volume")),
            open_interest=_int_or_none(row.get("openInterest")),
            raw_iv=_float_or_none(row.get("impliedVolatility")),
            computed_iv=_float_or_none(row.get("computedIV")),
            delta=_float_or_none(row.get("delta")),
            gamma=_float_or_none(row.get("gamma")),
            theta=_float_or_none(row.get("theta")),
            vega=_float_or_none(row.get("vega")),
            rho=_float_or_none(row.get("rho")),
            quote_timestamp=quote_ts.to_pydatetime() if pd.notna(quote_ts) else None,
            moneyness=_float_or_none(row.get("moneyness")),
            bid_ask_spread=_float_or_none(row.get("bidAskSpread")),
            bid_ask_spread_pct=_float_or_none(row.get("bidAskSpreadPct")),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a dashboard-compatible dictionary."""
        return {
            "contractSymbol": self.contract,
            "type": self.type,
            "expiration": self.expiry,
            "daysToExpiration": self.dte,
            "strike": self.strike,
            "moneyness": self.moneyness,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "last": self.last,
            "volume": self.volume,
            "openInterest": self.open_interest,
            "impliedVolatility": self.raw_iv,
            "computedIV": self.computed_iv,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "quoteTimestamp": self.quote_timestamp,
            "bidAskSpread": self.bid_ask_spread,
            "bidAskSpreadPct": self.bid_ask_spread_pct,
            "time_to_expiry": self.dte / 365.0 if self.dte else np.nan,
        }


@dataclass(frozen=True)
class MarketDataSnapshot:
    """Canonical symbol-level market snapshot."""

    symbol: str
    spot: float
    spot_timestamp: datetime
    chain_timestamp: Optional[datetime]
    expirations: tuple[datetime, ...] = field(default_factory=tuple)
    options: tuple[OptionQuote, ...] = field(default_factory=tuple)
    source: str = "unknown"
    source_delay: Optional[timedelta] = None
    cache_age: Optional[timedelta] = None
    fallback_reason: Optional[str] = None
    mode: str = "Unknown"
    raw_rows: int = 0
    valid_rows: int = 0
    rejected_rows: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_chain_frame(
        cls,
        symbol: str,
        spot: float,
        spot_timestamp: datetime,
        chain: pd.DataFrame,
        metadata: dict[str, Any],
    ) -> "MarketDataSnapshot":
        quotes = tuple(option_quotes_from_frame(chain))
        expirations = tuple(sorted({quote.expiry for quote in quotes if quote.expiry != datetime.min}))
        cache_age_seconds = metadata.get("cache_age_seconds")
        return cls(
            symbol=symbol.upper(),
            spot=float(spot),
            spot_timestamp=spot_timestamp,
            chain_timestamp=_datetime_or_none(metadata.get("timestamp")),
            expirations=expirations,
            options=quotes,
            source=str(metadata.get("source") or "unknown"),
            source_delay=_source_delay(metadata.get("mode")),
            cache_age=timedelta(seconds=int(cache_age_seconds)) if cache_age_seconds is not None else None,
            fallback_reason=metadata.get("fallback_reason"),
            mode=str(metadata.get("mode") or "Unknown"),
            raw_rows=int(metadata.get("raw_rows") or 0),
            valid_rows=int(metadata.get("valid_rows") or len(quotes)),
            rejected_rows=int(metadata.get("rejected_rows") or 0),
            warnings=tuple(str(item) for item in metadata.get("warnings") or ()),
        )

    def options_frame(self) -> pd.DataFrame:
        """Return options as the dashboard-compatible DataFrame shape."""
        return option_quotes_to_frame(self.options)

    def metadata_dict(self) -> dict[str, Any]:
        """Return provider metadata for diagnostics and captions."""
        return {
            "symbol": self.symbol,
            "source": self.source,
            "mode": self.mode,
            "timestamp": self.chain_timestamp,
            "expirations_requested": len(self.expirations),
            "expirations_loaded": len(self.expirations),
            "raw_rows": self.raw_rows,
            "valid_rows": self.valid_rows,
            "rejected_rows": self.rejected_rows,
            "cache_age_seconds": int(self.cache_age.total_seconds()) if self.cache_age is not None else None,
            "fallback_reason": self.fallback_reason,
            "warnings": list(self.warnings),
        }


def option_quotes_from_frame(frame: pd.DataFrame) -> list[OptionQuote]:
    """Build canonical quotes from a normalized option-chain DataFrame."""
    if frame.empty:
        return []
    return [OptionQuote.from_series(row) for _, row in frame.iterrows()]


def option_quotes_to_frame(quotes: Iterable[OptionQuote]) -> pd.DataFrame:
    """Build a normalized option-chain DataFrame from canonical quotes."""
    rows = [quote.as_dict() for quote in quotes]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(["expiration", "strike", "type"]).reset_index(drop=True)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _datetime_or_none(value: Any) -> Optional[datetime]:
    converted = pd.to_datetime(value, errors="coerce")
    return converted.to_pydatetime() if pd.notna(converted) else None


def _source_delay(mode: Any) -> Optional[timedelta]:
    text = str(mode or "").lower()
    if "delayed" in text:
        return timedelta(minutes=15)
    return None
