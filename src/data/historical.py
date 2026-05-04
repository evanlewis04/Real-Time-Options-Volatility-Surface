"""Historical price loading and realized-volatility calculations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.data.price_provider import YFINANCE_AVAILABLE
from src.data.retry import DataProviderRetryError, call_with_backoff


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricalPriceResult:
    """Normalized historical price result."""

    symbol: str
    frame: pd.DataFrame
    source: str
    timestamp: datetime
    period: str
    fallback_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return not self.frame.empty and "Close" in self.frame

    def close(self) -> pd.Series:
        return pd.to_numeric(self.frame["Close"], errors="coerce").dropna() if self.available else pd.Series(dtype=float)

    def returns(self) -> pd.Series:
        return self.close().pct_change().dropna()

    def realized_vol(self, window: int) -> pd.Series:
        return self.returns().rolling(window).std() * np.sqrt(252)


HistoryFetcher = Callable[[str, str], pd.DataFrame]


class HistoricalPriceLoader:
    """Load historical prices with cache and bounded yfinance retries."""

    def __init__(self, fetcher: HistoryFetcher | None = None):
        self.fetcher = fetcher or self._fetch_yfinance
        self.cache: Dict[tuple[str, str], HistoricalPriceResult] = {}

    def load(self, symbol: str, period: str = "1y") -> HistoricalPriceResult:
        key = (symbol.upper(), period)
        if key in self.cache:
            return self.cache[key]

        try:
            frame = self.fetcher(key[0], period)
            if frame.empty or "Close" not in frame:
                result = HistoricalPriceResult(
                    symbol=key[0],
                    frame=pd.DataFrame(),
                    source="yfinance historical closes",
                    timestamp=datetime.now(),
                    period=period,
                    fallback_reason="No historical closes returned",
                )
            else:
                result = HistoricalPriceResult(
                    symbol=key[0],
                    frame=frame.copy(),
                    source="yfinance historical closes",
                    timestamp=datetime.now(),
                    period=period,
                )
        except Exception as exc:
            logger.debug("Historical load failed for %s: %s", key[0], exc)
            result = HistoricalPriceResult(
                symbol=key[0],
                frame=pd.DataFrame(),
                source="yfinance historical closes",
                timestamp=datetime.now(),
                period=period,
                fallback_reason=str(exc),
            )

        self.cache[key] = result
        return result

    def load_many_returns(self, symbols: Iterable[str], period: str = "6mo", min_points: int = 20) -> Dict[str, pd.Series]:
        returns: Dict[str, pd.Series] = {}
        for symbol in symbols:
            result = self.load(symbol, period)
            series = result.returns()
            if len(series) >= min_points:
                returns[result.symbol] = series.rename(result.symbol)
        return returns

    def clear_cache(self) -> None:
        self.cache.clear()

    @staticmethod
    def _fetch_yfinance(symbol: str, period: str) -> pd.DataFrame:
        if not YFINANCE_AVAILABLE:
            raise DataProviderRetryError("yfinance is not installed")

        import yfinance as yf

        return call_with_backoff(
            lambda: yf.Ticker(symbol).history(period=period, auto_adjust=True),
            label=f"yfinance history {symbol}",
        )
