"""
Option-chain providers with explicit data provenance.

The dashboard uses this module to keep real market data separate from demo or
fallback data. A provider returns both a normalized options DataFrame and
metadata describing source, freshness, rejection counts, and fallback reasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.retry import call_with_backoff

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class OptionsChainMetadata:
    """Provenance and quality metadata for an option-chain snapshot."""

    symbol: str
    source: str
    mode: str
    timestamp: datetime
    expirations_requested: int = 0
    expirations_loaded: int = 0
    raw_rows: int = 0
    valid_rows: int = 0
    rejected_rows: int = 0
    median_spread_pct: Optional[float] = None
    stale_quote_count: int = 0
    last_only_quote_count: int = 0
    zero_bid_ask_count: int = 0
    stale_last_only_rejected_count: int = 0
    max_quote_age_days: int = 5
    fallback_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        out = self.__dict__.copy()
        out["timestamp"] = self.timestamp
        return out


class YFinanceOptionsProvider:
    """Fetch and normalize delayed option-chain data from yfinance."""

    def __init__(self, max_expirations: int = 8, cache_ttl_seconds: int = 300, max_quote_age_days: int = 5):
        self.max_expirations = max_expirations
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_quote_age_days = max_quote_age_days
        self.expiration_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, datetime]] = {}

    def fetch_chain(self, symbol: str, spot_price: float) -> Tuple[pd.DataFrame, OptionsChainMetadata]:
        now = datetime.now()
        key = symbol.upper()
        meta = OptionsChainMetadata(
            symbol=key,
            source="yfinance",
            mode="Live/Delayed",
            timestamp=now,
            max_quote_age_days=self.max_quote_age_days,
        )

        if not YFINANCE_AVAILABLE:
            meta.mode = "Unavailable"
            meta.fallback_reason = "yfinance is not installed"
            return pd.DataFrame(), meta

        try:
            ticker = yf.Ticker(key)
            expirations = list(
                call_with_backoff(
                    lambda: ticker.options or [],
                    label=f"yfinance expirations {key}",
                )
            )
            meta.expirations_requested = min(len(expirations), self.max_expirations)
            if not expirations:
                meta.mode = "Unavailable"
                meta.fallback_reason = "No option expirations returned by yfinance"
                return pd.DataFrame(), meta

            frames = []
            for expiration in expirations[: self.max_expirations]:
                try:
                    frames.append(self._fetch_expiration_frame(ticker, key, expiration, now))
                    meta.expirations_loaded += 1
                except Exception as exc:
                    meta.warnings.append(f"{expiration}: {exc}")
                    logger.debug("Failed to load %s %s options: %s", key, expiration, exc)

            if not frames:
                meta.mode = "Unavailable"
                meta.fallback_reason = "All yfinance option expiration loads failed"
                return pd.DataFrame(), meta

            raw = pd.concat(frames, ignore_index=True)
            meta.raw_rows = len(raw)
            clean = self._normalize(raw, key, spot_price, now, max_quote_age_days=self.max_quote_age_days)
            meta.valid_rows = len(clean)
            meta.rejected_rows = max(0, meta.raw_rows - meta.valid_rows)
            meta.stale_quote_count = int(clean.attrs.get("stale_quote_count", 0))
            meta.last_only_quote_count = int(clean.attrs.get("last_only_quote_count", 0))
            meta.zero_bid_ask_count = int(clean.attrs.get("zero_bid_ask_count", 0))
            meta.stale_last_only_rejected_count = int(clean.attrs.get("stale_last_only_rejected_count", 0))
            if "bidAskSpreadPct" in clean and not clean.empty:
                meta.median_spread_pct = float(clean["bidAskSpreadPct"].median())

            if clean.empty:
                meta.mode = "Unavailable"
                meta.fallback_reason = "No valid option rows after quality filters"
            return clean, meta
        except Exception as exc:
            meta.mode = "Unavailable"
            meta.fallback_reason = str(exc)
            logger.warning("yfinance chain fetch failed for %s: %s", key, exc)
            return pd.DataFrame(), meta

    def clear_cache(self) -> None:
        """Clear cached expiration-level yfinance chains."""
        self.expiration_cache.clear()

    def cache_status(self) -> Dict[str, Any]:
        """Return cache status for diagnostics."""
        return {
            "entries": len(self.expiration_cache),
            "ttl_seconds": int(self.cache_ttl.total_seconds()),
            "keys": [f"{symbol}:{expiration}" for symbol, expiration in sorted(self.expiration_cache)],
        }

    def _fetch_expiration_frame(self, ticker: Any, symbol: str, expiration: str, now: datetime) -> pd.DataFrame:
        cache_key = (symbol, expiration)
        cached = self.expiration_cache.get(cache_key)
        if cached and now - cached[1] < self.cache_ttl:
            return cached[0].copy()

        chain = call_with_backoff(
            lambda: ticker.option_chain(expiration),
            label=f"yfinance option_chain {symbol} {expiration}",
        )
        calls = chain.calls.copy()
        calls["type"] = "call"
        calls["expiration"] = expiration
        puts = chain.puts.copy()
        puts["type"] = "put"
        puts["expiration"] = expiration
        frame = pd.concat([calls, puts], ignore_index=True)
        self.expiration_cache[cache_key] = (frame.copy(), now)
        return frame

    @staticmethod
    def _normalize(
        raw: pd.DataFrame,
        symbol: str,
        spot_price: float,
        now: datetime,
        max_quote_age_days: int = 5,
    ) -> pd.DataFrame:
        df = raw.copy()
        df["symbol"] = symbol
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")

        today = now.date()
        df["daysToExpiration"] = df["expiration"].apply(
            lambda value: (value.date() - today).days if pd.notna(value) else np.nan
        )
        df["time_to_expiry"] = df["daysToExpiration"] / 365.0
        df["moneyness"] = df["strike"] / spot_price

        numeric_cols = [
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "volume",
            "openInterest",
            "impliedVolatility",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "lastPrice" in df.columns:
            df = df.rename(columns={"lastPrice": "last"})
        if "lastTradeDate" in df.columns:
            df = df.rename(columns={"lastTradeDate": "quoteTimestamp"})

        for col in ("bid", "ask", "last", "volume", "openInterest", "impliedVolatility"):
            if col not in df.columns:
                df[col] = np.nan
        if "quoteTimestamp" not in df.columns:
            df["quoteTimestamp"] = pd.NaT

        quote_ts = pd.to_datetime(df["quoteTimestamp"], errors="coerce", utc=True).dt.tz_convert(None)
        quote_age = now - quote_ts
        max_quote_age = timedelta(days=max_quote_age_days)
        df["quoteAgeSeconds"] = quote_age.dt.total_seconds()
        df.loc[quote_ts.isna(), "quoteAgeSeconds"] = np.nan
        df["isStaleQuote"] = quote_ts.notna() & (quote_age > max_quote_age)

        valid_bid_ask = (df["bid"] > 0) & (df["ask"] > df["bid"])
        last_only = ~valid_bid_ask & (df["last"] > 0)
        zero_bid_ask = df["bid"].fillna(0) <= 0
        zero_bid_ask &= df["ask"].fillna(0) <= 0
        df["quoteQuality"] = np.select(
            [
                valid_bid_ask & df["isStaleQuote"],
                valid_bid_ask,
                last_only & df["isStaleQuote"],
                last_only,
            ],
            ["stale_bid_ask", "bid_ask", "stale_last_only", "last_only"],
            default="invalid",
        )
        df["mid"] = np.where(
            valid_bid_ask,
            (df["bid"] + df["ask"]) / 2,
            df["last"],
        )
        df["bidAskSpread"] = df["ask"] - df["bid"]
        df["bidAskSpreadPct"] = df["bidAskSpread"] / df["mid"].replace(0, np.nan)

        clean = df[
            (df["strike"] > 0)
            & (df["daysToExpiration"] > 0)
            & (df["time_to_expiry"] > 0)
            & (df["mid"] > 0)
            & (df["impliedVolatility"] > 0.01)
            & (df["impliedVolatility"] < 5.0)
            & (df["moneyness"] > 0.35)
            & (df["moneyness"] < 2.5)
        ].copy()

        liquid = (
            ((clean["bid"] > 0) & (clean["ask"] > clean["bid"]) & (clean["bidAskSpreadPct"] < 1.5))
            | clean["bidAskSpreadPct"].isna()
        )
        fresh_enough = clean["quoteQuality"] != "stale_last_only"
        clean = clean[liquid].copy()
        stale_last_only_rejected = int((~fresh_enough & liquid).sum())
        clean = clean[clean["quoteQuality"] != "stale_last_only"].copy()
        clean.attrs["stale_quote_count"] = int(clean["isStaleQuote"].sum()) if "isStaleQuote" in clean else 0
        clean.attrs["last_only_quote_count"] = int((clean["quoteQuality"] == "last_only").sum())
        clean.attrs["zero_bid_ask_count"] = int(zero_bid_ask.sum())
        clean.attrs["stale_last_only_rejected_count"] = stale_last_only_rejected

        ordered_cols = [
            "symbol",
            "contractSymbol",
            "type",
            "expiration",
            "daysToExpiration",
            "strike",
            "moneyness",
            "bid",
            "ask",
            "mid",
            "last",
            "volume",
            "openInterest",
            "impliedVolatility",
            "bidAskSpread",
            "bidAskSpreadPct",
            "quoteQuality",
            "isStaleQuote",
            "quoteAgeSeconds",
            "quoteTimestamp",
            "time_to_expiry",
        ]
        available = [col for col in ordered_cols if col in clean.columns]
        result = clean[available].sort_values(["expiration", "strike", "type"]).reset_index(drop=True)
        result.attrs.update(clean.attrs)
        return result
