"""
Option-chain providers with explicit data provenance.

The dashboard uses this module to keep real market data separate from demo or
fallback data. A provider returns both a normalized options DataFrame and
metadata describing source, freshness, rejection counts, and fallback reasons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
    fallback_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        out = self.__dict__.copy()
        out["timestamp"] = self.timestamp
        return out


class YFinanceOptionsProvider:
    """Fetch and normalize delayed option-chain data from yfinance."""

    def __init__(self, max_expirations: int = 8):
        self.max_expirations = max_expirations

    def fetch_chain(self, symbol: str, spot_price: float) -> Tuple[pd.DataFrame, OptionsChainMetadata]:
        now = datetime.now()
        key = symbol.upper()
        meta = OptionsChainMetadata(
            symbol=key,
            source="yfinance",
            mode="Live/Delayed",
            timestamp=now,
        )

        if not YFINANCE_AVAILABLE:
            meta.mode = "Unavailable"
            meta.fallback_reason = "yfinance is not installed"
            return pd.DataFrame(), meta

        try:
            ticker = yf.Ticker(key)
            expirations = list(ticker.options or [])
            meta.expirations_requested = min(len(expirations), self.max_expirations)
            if not expirations:
                meta.mode = "Unavailable"
                meta.fallback_reason = "No option expirations returned by yfinance"
                return pd.DataFrame(), meta

            frames = []
            for expiration in expirations[: self.max_expirations]:
                try:
                    chain = ticker.option_chain(expiration)
                    calls = chain.calls.copy()
                    calls["type"] = "call"
                    calls["expiration"] = expiration
                    puts = chain.puts.copy()
                    puts["type"] = "put"
                    puts["expiration"] = expiration
                    frames.append(pd.concat([calls, puts], ignore_index=True))
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
            clean = self._normalize(raw, key, spot_price, now)
            meta.valid_rows = len(clean)
            meta.rejected_rows = max(0, meta.raw_rows - meta.valid_rows)
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

    @staticmethod
    def _normalize(raw: pd.DataFrame, symbol: str, spot_price: float, now: datetime) -> pd.DataFrame:
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

        df["mid"] = np.where(
            (df["bid"] > 0) & (df["ask"] > df["bid"]),
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
        clean = clean[liquid].copy()

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
            "quoteTimestamp",
            "time_to_expiry",
        ]
        available = [col for col in ordered_cols if col in clean.columns]
        return clean[available].sort_values(["expiration", "strike", "type"]).reset_index(drop=True)
