"""
Real-time stock price provider with yfinance, in-memory caching, and a
simulated-walk fallback for when the live feed is unavailable.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed; price provider will run in simulated mode")


# Static fallback prices (last known good levels); used when yfinance is offline
# and as the base for the simulated random walk.
FALLBACK_PRICES: Dict[str, float] = {
    'AAPL': 196.50, 'MSFT': 416.50, 'GOOGL': 166.80,
    'META': 540.00, 'AMZN': 177.50, 'NVDA': 138.50, 'TSLA': 325.00,
    'AMD': 132.00, 'NFLX': 720.00, 'CRM': 285.00,
    'ORCL': 115.00, 'ADBE': 565.00, 'PLTR': 62.00,
    'INTC': 25.00, 'IBM': 235.00, 'CSCO': 58.00,
    'SPY': 578.00, 'QQQ': 478.00, 'IWM': 215.00, 'VTI': 275.00,
    'JPM': 241.00, 'BAC': 42.50, 'WFC': 68.00, 'GS': 485.00,
    'JNJ': 155.00, 'PFE': 26.50, 'UNH': 590.00, 'MRNA': 85.00,
    'KO': 59.50, 'PEP': 165.00, 'WMT': 185.00, 'HD': 415.00,
    'DIS': 95.00, 'SPOT': 425.00,
    'COIN': 245.00, 'SQ': 85.00, 'PYPL': 62.00,
    'GME': 20.50, 'AMC': 4.25, 'RBLX': 45.00,
    'UBER': 68.50, 'LYFT': 16.50, 'F': 11.00, 'GM': 42.00,
    'XOM': 115.00, 'CVX': 158.00, 'COP': 108.00,
    'V': 295.00, 'MA': 485.00, 'BABA': 78.00,
    'NIO': 4.50, 'RIVN': 12.00, 'LCID': 2.80,
    'SOFI': 16.00, 'HOOD': 28.00, 'DKNG': 42.00,
}

# Per-symbol intraday vol regimes for the simulated walk.
_HIGH_VOL_TICKERS = {'TSLA', 'GME', 'PLTR', 'NVDA'}
_ETF_TICKERS = {'SPY', 'QQQ', 'VTI'}
_MEGACAP_TICKERS = {'AAPL', 'MSFT', 'GOOGL'}


class RealTimePriceProvider:
    """Live prices via yfinance with caching and a simulated-walk fallback."""

    def __init__(self, cache_duration_seconds: int = 60):
        self.cache_duration = cache_duration_seconds
        self.price_cache: Dict[str, float] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.price_movements: Dict[str, float] = {sym: 0.0 for sym in FALLBACK_PRICES}
        self.last_movement_update = datetime.now()
        self.current_market_prices = dict(FALLBACK_PRICES)
        self.yfinance_working = self._test_yfinance()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_live_price(self, symbol: str) -> float:
        """Return the most recent price for ``symbol``; cached for ``cache_duration``."""
        now = datetime.now()
        key = symbol.upper()

        cached = self.price_cache.get(key)
        ts = self.cache_timestamps.get(key)
        if cached is not None and ts and (now - ts).total_seconds() < self.cache_duration:
            return cached

        if self.yfinance_working:
            live = self._fetch_yfinance_price(key)
            if live is not None:
                self._cache(key, live, now)
                return live

        sim = self._simulated_price(key)
        self._cache(key, sim, now)
        return sim

    def clear_cache(self) -> None:
        self.price_cache.clear()
        self.cache_timestamps.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache(self, key: str, price: float, when: datetime) -> None:
        self.price_cache[key] = price
        self.cache_timestamps[key] = when

    def _test_yfinance(self) -> bool:
        if not YFINANCE_AVAILABLE:
            return False
        try:
            data = yf.Ticker("AAPL").history(period="1d")
            return not data.empty and float(data['Close'].iloc[-1]) > 0
        except Exception as e:
            logger.warning(f"yfinance smoke test failed: {e}")
            return False

    def _fetch_yfinance_price(self, symbol: str) -> Optional[float]:
        """Try several yfinance endpoints in order of freshness; return None if all fail."""
        try:
            ticker = yf.Ticker(symbol)

            for kwargs in ({"period": "1d", "interval": "1m"}, {"period": "2d"}):
                hist = ticker.history(**kwargs)
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    if price > 0:
                        return price

            try:
                fast = ticker.fast_info
                last = getattr(fast, 'last_price', None)
                if last and last > 0:
                    return float(last)
            except (AttributeError, ValueError, TypeError, OSError):
                pass

            info = ticker.info or {}
            for field in ('currentPrice', 'regularMarketPrice', 'previousClose'):
                value = info.get(field)
                if isinstance(value, (int, float)) and value > 0:
                    return float(value)

        except Exception as e:
            logger.debug(f"yfinance fetch failed for {symbol}: {e}")

        return None

    def _simulated_price(self, symbol: str) -> float:
        """Random-walk simulated price around the fallback level."""
        self._update_price_movements()
        base = self.current_market_prices.get(symbol, 100.0)
        drift = self.price_movements.get(symbol, 0.0)
        noise = np.random.normal(0, 0.002)
        price = base * (1 + drift + noise)
        price = float(np.clip(price, base * 0.85, base * 1.15))
        return round(price, 2)

    def _update_price_movements(self) -> None:
        """Step the random-walk drift for all symbols once per 30s."""
        now = datetime.now()
        if (now - self.last_movement_update).total_seconds() < 30:
            return
        self.last_movement_update = now

        market_hours = 9 <= now.hour <= 16
        for symbol in self.price_movements:
            if symbol in _HIGH_VOL_TICKERS:
                daily_vol = 0.04 if market_hours else 0.01
            elif symbol in _ETF_TICKERS:
                daily_vol = 0.015 if market_hours else 0.005
            elif symbol in _MEGACAP_TICKERS:
                daily_vol = 0.025 if market_hours else 0.008
            else:
                daily_vol = 0.03 if market_hours else 0.01

            shock = np.random.normal(0, daily_vol / 48)
            mean_reversion = -0.1 * self.price_movements[symbol]
            self.price_movements[symbol] = float(
                np.clip(self.price_movements[symbol] + shock + mean_reversion, -0.08, 0.08)
            )
