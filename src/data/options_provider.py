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
    min_open_interest: int = 0
    min_volume: int = 0
    max_bid_ask_spread_pct: float = 1.5
    stale_quote_count: int = 0
    last_only_quote_count: int = 0
    zero_bid_ask_count: int = 0
    crossed_market_count: int = 0
    locked_market_count: int = 0
    crossed_locked_rejected_count: int = 0
    stale_last_only_rejected_count: int = 0
    liquidity_filtered_count: int = 0
    low_open_interest_rejected_count: int = 0
    low_volume_rejected_count: int = 0
    wide_spread_rejected_count: int = 0
    old_quote_rejected_count: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    data_quality_score: Optional[float] = None
    expiry_quality: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_quote_age_days: int = 5
    fallback_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        out = self.__dict__.copy()
        out["timestamp"] = self.timestamp
        return out


class YFinanceOptionsProvider:
    """Fetch and normalize delayed option-chain data from yfinance."""

    def __init__(
        self,
        max_expirations: int = 8,
        cache_ttl_seconds: int = 300,
        max_quote_age_days: int = 5,
        min_open_interest: int = 0,
        min_volume: int = 0,
        max_bid_ask_spread_pct: float = 1.5,
    ):
        self.max_expirations = max_expirations
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_quote_age_days = max_quote_age_days
        self.min_open_interest = max(0, int(min_open_interest))
        self.min_volume = max(0, int(min_volume))
        self.max_bid_ask_spread_pct = max(0.0, float(max_bid_ask_spread_pct))
        self.expiration_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, datetime]] = {}

    def configure_liquidity_filters(
        self,
        *,
        min_open_interest: Optional[int] = None,
        min_volume: Optional[int] = None,
        max_bid_ask_spread_pct: Optional[float] = None,
        max_quote_age_days: Optional[int] = None,
    ) -> bool:
        """Update provider-level liquidity filters.

        Returns ``True`` when any setting changed so callers can invalidate
        chain caches that were normalized with the old policy.
        """
        old = self.liquidity_filter_settings()
        if min_open_interest is not None:
            self.min_open_interest = max(0, int(min_open_interest))
        if min_volume is not None:
            self.min_volume = max(0, int(min_volume))
        if max_bid_ask_spread_pct is not None:
            self.max_bid_ask_spread_pct = max(0.0, float(max_bid_ask_spread_pct))
        if max_quote_age_days is not None:
            self.max_quote_age_days = max(0, int(max_quote_age_days))
        return old != self.liquidity_filter_settings()

    def liquidity_filter_settings(self) -> Dict[str, Any]:
        """Return active liquidity-filter settings."""
        return {
            "min_open_interest": self.min_open_interest,
            "min_volume": self.min_volume,
            "max_bid_ask_spread_pct": self.max_bid_ask_spread_pct,
            "max_quote_age_days": self.max_quote_age_days,
        }

    def fetch_chain(self, symbol: str, spot_price: float) -> Tuple[pd.DataFrame, OptionsChainMetadata]:
        now = datetime.now()
        key = symbol.upper()
        meta = OptionsChainMetadata(
            symbol=key,
            source="yfinance",
            mode="Live/Delayed",
            timestamp=now,
            max_quote_age_days=self.max_quote_age_days,
            min_open_interest=self.min_open_interest,
            min_volume=self.min_volume,
            max_bid_ask_spread_pct=self.max_bid_ask_spread_pct,
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
            clean = self._normalize(
                raw,
                key,
                spot_price,
                now,
                max_quote_age_days=self.max_quote_age_days,
                min_open_interest=self.min_open_interest,
                min_volume=self.min_volume,
                max_bid_ask_spread_pct=self.max_bid_ask_spread_pct,
            )
            meta.valid_rows = len(clean)
            meta.rejected_rows = max(0, meta.raw_rows - meta.valid_rows)
            meta.stale_quote_count = int(clean.attrs.get("stale_quote_count", 0))
            meta.last_only_quote_count = int(clean.attrs.get("last_only_quote_count", 0))
            meta.zero_bid_ask_count = int(clean.attrs.get("zero_bid_ask_count", 0))
            meta.crossed_market_count = int(clean.attrs.get("crossed_market_count", 0))
            meta.locked_market_count = int(clean.attrs.get("locked_market_count", 0))
            meta.crossed_locked_rejected_count = int(clean.attrs.get("crossed_locked_rejected_count", 0))
            meta.stale_last_only_rejected_count = int(clean.attrs.get("stale_last_only_rejected_count", 0))
            meta.liquidity_filtered_count = int(clean.attrs.get("liquidity_filtered_count", 0))
            meta.low_open_interest_rejected_count = int(clean.attrs.get("low_open_interest_rejected_count", 0))
            meta.low_volume_rejected_count = int(clean.attrs.get("low_volume_rejected_count", 0))
            meta.wide_spread_rejected_count = int(clean.attrs.get("wide_spread_rejected_count", 0))
            meta.old_quote_rejected_count = int(clean.attrs.get("old_quote_rejected_count", 0))
            meta.rejection_reasons = dict(clean.attrs.get("rejection_reasons", {}))
            meta.data_quality_score = clean.attrs.get("data_quality_score")
            meta.expiry_quality = dict(clean.attrs.get("expiry_quality", {}))
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
        min_open_interest: int = 0,
        min_volume: int = 0,
        max_bid_ask_spread_pct: float = 1.5,
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
            "mark",
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

        for col in ("bid", "ask", "last", "mark", "volume", "openInterest", "impliedVolatility"):
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

        positive_bid_ask = (df["bid"] > 0) & (df["ask"] > 0)
        crossed_market = positive_bid_ask & (df["bid"] > df["ask"])
        locked_market = positive_bid_ask & (df["bid"] == df["ask"])
        crossed_locked_market = crossed_market | locked_market
        df["isCrossedMarket"] = crossed_market
        df["isLockedMarket"] = locked_market
        valid_bid_ask = positive_bid_ask & (df["ask"] > df["bid"])
        last_only = ~valid_bid_ask & ~crossed_locked_market & (df["last"] > 0)
        zero_bid_ask = df["bid"].fillna(0) <= 0
        zero_bid_ask &= df["ask"].fillna(0) <= 0
        df["quoteQuality"] = np.select(
            [
                crossed_market,
                locked_market,
                valid_bid_ask & df["isStaleQuote"],
                valid_bid_ask,
                last_only & df["isStaleQuote"],
                last_only,
            ],
            ["crossed_market", "locked_market", "stale_bid_ask", "bid_ask", "stale_last_only", "last_only"],
            default="invalid",
        )
        df["mid"] = np.where(
            valid_bid_ask,
            (df["bid"] + df["ask"]) / 2,
            np.nan,
        )
        provider_mark = pd.to_numeric(df["mark"], errors="coerce")
        df["markSource"] = np.select(
            [
                provider_mark > 0,
                valid_bid_ask,
                last_only,
            ],
            ["provider_mark", "midpoint", "last"],
            default="unavailable",
        )
        df["mark"] = np.where(
            provider_mark > 0,
            provider_mark,
            np.where(valid_bid_ask, df["mid"], df["last"]),
        )
        df["bidAskSpread"] = df["ask"] - df["bid"]
        df["bidAskSpreadPct"] = df["bidAskSpread"] / df["mid"].replace(0, np.nan)
        expiry_raw_counts = _expiry_counts(df)
        expiry_rejection_reasons: Dict[str, Dict[str, int]] = {}

        base_mask = (
            (df["strike"] > 0)
            & (df["daysToExpiration"] > 0)
            & (df["time_to_expiry"] > 0)
            & (df["mark"] > 0)
            & (df["impliedVolatility"] > 0.01)
            & (df["impliedVolatility"] < 5.0)
            & (df["moneyness"] > 0.35)
            & (df["moneyness"] < 2.5)
        )
        _record_expiry_rejections(df, base_mask, "base_quality", expiry_rejection_reasons)
        clean = df[base_mask].copy()
        rejection_reasons: Dict[str, int] = {
            "base_quality": int((~base_mask).sum()),
        }

        clean, crossed_locked_rejected = _apply_row_filter(
            clean,
            ~(clean["isCrossedMarket"] | clean["isLockedMarket"]),
            "crossed_locked_market",
            rejection_reasons,
            expiry_rejection_reasons,
        )
        clean, stale_last_only_rejected = _apply_row_filter(
            clean,
            clean["quoteQuality"] != "stale_last_only",
            "stale_last_only",
            rejection_reasons,
            expiry_rejection_reasons,
        )
        clean, old_quote_rejected = _apply_row_filter(
            clean,
            ~(pd.to_numeric(clean["quoteAgeSeconds"], errors="coerce") > max_quote_age.total_seconds()),
            "old_quote",
            rejection_reasons,
            expiry_rejection_reasons,
        )
        clean, low_oi_rejected = _apply_row_filter(
            clean,
            pd.to_numeric(clean["openInterest"], errors="coerce").fillna(0) >= max(0, int(min_open_interest)),
            "low_open_interest",
            rejection_reasons,
            expiry_rejection_reasons,
        )
        clean, low_volume_rejected = _apply_row_filter(
            clean,
            pd.to_numeric(clean["volume"], errors="coerce").fillna(0) >= max(0, int(min_volume)),
            "low_volume",
            rejection_reasons,
            expiry_rejection_reasons,
        )
        spread_pct = pd.to_numeric(clean["bidAskSpreadPct"], errors="coerce")
        spread_ok = spread_pct.isna() | (spread_pct <= max(0.0, float(max_bid_ask_spread_pct)))
        clean, wide_spread_rejected = _apply_row_filter(
            clean,
            spread_ok,
            "wide_bid_ask_spread",
            rejection_reasons,
            expiry_rejection_reasons,
        )
        liquidity_filtered_count = (
            stale_last_only_rejected
            + old_quote_rejected
            + low_oi_rejected
            + low_volume_rejected
            + wide_spread_rejected
        )
        clean.attrs["stale_quote_count"] = int(clean["isStaleQuote"].sum()) if "isStaleQuote" in clean else 0
        clean.attrs["last_only_quote_count"] = int((clean["quoteQuality"] == "last_only").sum())
        clean.attrs["zero_bid_ask_count"] = int(zero_bid_ask.sum())
        clean.attrs["crossed_market_count"] = int(crossed_market.sum())
        clean.attrs["locked_market_count"] = int(locked_market.sum())
        clean.attrs["crossed_locked_rejected_count"] = crossed_locked_rejected
        clean.attrs["stale_last_only_rejected_count"] = stale_last_only_rejected
        clean.attrs["old_quote_rejected_count"] = old_quote_rejected
        clean.attrs["low_open_interest_rejected_count"] = low_oi_rejected
        clean.attrs["low_volume_rejected_count"] = low_volume_rejected
        clean.attrs["wide_spread_rejected_count"] = wide_spread_rejected
        clean.attrs["liquidity_filtered_count"] = liquidity_filtered_count
        clean.attrs["rejection_reasons"] = {key: value for key, value in rejection_reasons.items() if value}
        clean.attrs["expiry_quality"] = _expiry_quality_summary(clean, expiry_raw_counts, expiry_rejection_reasons)
        clean.attrs["data_quality_score"] = _quality_score(len(clean), int(sum(rejection_reasons.values())), clean.attrs["rejection_reasons"])

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
            "mark",
            "markSource",
            "last",
            "volume",
            "openInterest",
            "impliedVolatility",
            "bidAskSpread",
            "bidAskSpreadPct",
            "quoteQuality",
            "isCrossedMarket",
            "isLockedMarket",
            "isStaleQuote",
            "quoteAgeSeconds",
            "quoteTimestamp",
            "time_to_expiry",
        ]
        available = [col for col in ordered_cols if col in clean.columns]
        result = clean[available].sort_values(["expiration", "strike", "type"]).reset_index(drop=True)
        result.attrs.update(clean.attrs)
        return result


def _apply_row_filter(
    frame: pd.DataFrame,
    keep_mask: pd.Series,
    reason: str,
    rejection_reasons: Dict[str, int],
    expiry_rejection_reasons: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[pd.DataFrame, int]:
    """Apply one sequential row filter and track its rejection reason."""
    if frame.empty:
        rejection_reasons[reason] = 0
        return frame.copy(), 0
    aligned = keep_mask.reindex(frame.index).fillna(False).astype(bool)
    if expiry_rejection_reasons is not None:
        _record_expiry_rejections(frame, aligned, reason, expiry_rejection_reasons)
    rejected = int((~aligned).sum())
    rejection_reasons[reason] = rejected
    return frame[aligned].copy(), rejected


def _expiry_key(value: Any) -> str:
    expiry = pd.to_datetime(value, errors="coerce")
    return expiry.date().isoformat() if pd.notna(expiry) else "unknown"


def _expiry_counts(frame: pd.DataFrame) -> Dict[str, int]:
    if frame.empty or "expiration" not in frame:
        return {}
    expirations = frame["expiration"].map(_expiry_key)
    return {str(expiry): int(count) for expiry, count in expirations.value_counts().items()}


def _record_expiry_rejections(
    frame: pd.DataFrame,
    keep_mask: pd.Series,
    reason: str,
    expiry_rejection_reasons: Dict[str, Dict[str, int]],
) -> None:
    if frame.empty or "expiration" not in frame:
        return
    aligned = keep_mask.reindex(frame.index).fillna(False).astype(bool)
    rejected = frame[~aligned]
    if rejected.empty:
        return
    for expiry, count in _expiry_counts(rejected).items():
        buckets = expiry_rejection_reasons.setdefault(expiry, {})
        buckets[reason] = buckets.get(reason, 0) + int(count)


def _expiry_quality_summary(
    clean: pd.DataFrame,
    expiry_raw_counts: Dict[str, int],
    expiry_rejection_reasons: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Any]]:
    valid_counts = _expiry_counts(clean)
    out: Dict[str, Dict[str, Any]] = {}
    for expiry in sorted(set(expiry_raw_counts) | set(valid_counts) | set(expiry_rejection_reasons)):
        reason_buckets = {
            reason: int(count)
            for reason, count in expiry_rejection_reasons.get(expiry, {}).items()
            if count
        }
        valid = int(valid_counts.get(expiry, 0))
        rejected = int(sum(reason_buckets.values()))
        raw = int(expiry_raw_counts.get(expiry, valid + rejected))
        out[expiry] = {
            "score": _quality_score(valid, rejected, reason_buckets),
            "raw_quotes": raw,
            "valid_quotes": valid,
            "rejected_quotes": rejected,
            "reason_buckets": reason_buckets,
        }
    return out


def _quality_score(valid_quotes: int, rejected_quotes: int, reason_buckets: Dict[str, int]) -> float:
    valid = max(0, int(valid_quotes or 0))
    rejected = max(0, int(rejected_quotes or 0))
    total = valid + rejected
    if total <= 0:
        return 0.0

    score = 100.0 * valid / total
    score -= 20.0 * int(reason_buckets.get("computed_iv_failed", 0)) / max(valid, 1)
    score -= 15.0 * int(reason_buckets.get("parity_violation", 0)) / max(valid, 1)
    return round(float(min(100.0, max(0.0, score))), 1)
