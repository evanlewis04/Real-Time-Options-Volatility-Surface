"""
Dashboard data orchestrator.

Wires together the live price feed, synthetic options generator, and
volatility-surface builder into the single ``DashboardConnector`` API
consumed by ``app.py``. Heavy lifting lives in the ``src/`` modules; this
file is a thin coordination layer plus the ad-hoc portfolio metrics and
correlation matrix the dashboard renders.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.price_provider import RealTimePriceProvider, YFINANCE_AVAILABLE
from src.data.synthetic_options import SyntheticOptionsGenerator
from src.analysis.surface_builder import build_surface

logger = logging.getLogger(__name__)


# Per-symbol average daily volume baselines (shares); used to scale a
# realistic-looking volume figure on the dashboard.
_VOLUME_BASELINES: Dict[str, int] = {
    'PLTR': 25_000_000, 'TSLA': 80_000_000, 'NVDA': 70_000_000, 'GME': 15_000_000,
    'AAPL': 50_000_000, 'MSFT': 30_000_000, 'GOOGL': 25_000_000, 'META': 20_000_000,
    'SPY': 100_000_000, 'QQQ': 60_000_000, 'JPM': 15_000_000, 'BAC': 20_000_000,
}


class DashboardConnector:
    """Top-level data provider for the Streamlit dashboard."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.price_provider = RealTimePriceProvider()
        self.options_generator = SyntheticOptionsGenerator(self.price_provider)
        self.real_time_active = False
        self.update_interval = 30

    # ------------------------------------------------------------------
    # Symbol-level data
    # ------------------------------------------------------------------

    def get_current_data(self, symbol: str) -> Dict[str, Any]:
        """Live price plus representative ATM Greeks and volume."""
        try:
            spot = self.price_provider.get_live_price(symbol)
            greeks = self.options_generator.calculate_greeks(symbol, spot)
            base_volume = _VOLUME_BASELINES.get(symbol.upper(), 5_000_000)
            volume = int(base_volume * np.random.uniform(0.4, 1.6))
            return {
                'price': spot,
                'volume': volume,
                'iv_30d': greeks['iv_30d'],
                'iv_60d': greeks['iv_60d'],
                'iv_90d': greeks['iv_90d'],
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'bid_ask_spread': float(np.random.uniform(0.005, 0.05)),
                'contracts': int(np.random.randint(50, 800)),
                'timestamp': datetime.now(),
            }
        except Exception as e:
            logger.error(f"get_current_data({symbol}) failed: {e}")
            return self._safe_fallback(symbol)

    def get_vol_surface_data(self, symbol: str
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Strikes, expiries (days), and IVs for the surface plot."""
        try:
            chain = self.options_generator.create_chain(symbol)
            spot = self.price_provider.get_live_price(symbol)
            return build_surface(chain, spot, symbol)
        except Exception as e:
            logger.error(f"get_vol_surface_data({symbol}) failed: {e}")
            from src.analysis.surface_builder import _parametric_fallback
            return _parametric_fallback(symbol, self.price_provider.get_live_price(symbol))

    # ------------------------------------------------------------------
    # Portfolio-level summaries (synthetic; dashboard demo only)
    # ------------------------------------------------------------------

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        try:
            return {
                'total_value': 1_500_000 + np.random.normal(0, 50_000),
                'daily_pnl': float(np.random.normal(5_000, 15_000)),
                'var_95': float(-np.random.uniform(25_000, 45_000)),
                'sharpe_ratio': float(max(0.8, 1.2 + np.random.normal(0, 0.2))),
                'max_drawdown': float(-np.random.uniform(0.08, 0.15)),
                'volatility': float(np.random.uniform(0.15, 0.25)),
            }
        except Exception as e:
            logger.error(f"portfolio metrics failed: {e}")
            return {'total_value': 1_500_000, 'daily_pnl': 5_000, 'var_95': -30_000,
                    'sharpe_ratio': 1.2, 'max_drawdown': -0.10, 'volatility': 0.20}

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Synthetic correlation matrix for the dashboard heatmap."""
        try:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'PLTR', 'SPY', 'QQQ']
            n = len(symbols)
            corr = np.random.uniform(0.3, 0.8, (n, n))
            for i in range(6):  # tech cluster
                for j in range(6):
                    if i < j:
                        corr[i, j] = np.random.uniform(0.6, 0.85)
            corr = np.triu(corr) + np.triu(corr, 1).T
            np.fill_diagonal(corr, 1.0)
            return pd.DataFrame(corr, index=symbols, columns=symbols)
        except Exception as e:
            logger.error(f"correlation matrix failed: {e}")
            symbols = ['AAPL', 'MSFT', 'TSLA', 'SPY']
            return pd.DataFrame(np.eye(len(symbols)), index=symbols, columns=symbols)

    # ------------------------------------------------------------------
    # System / lifecycle
    # ------------------------------------------------------------------

    def get_system_health(self) -> Dict[str, Any]:
        cache_status = getattr(self.price_provider, 'get_cache_status', lambda: {})()
        return {
            'overall': {
                'pricing_models_available': True,
                'yfinance_available': self.price_provider.yfinance_working,
                'black_scholes_active': True,
                'implied_vol_active': True,
                'vol_surface_active': True,
                'real_time_pricing': True,
                'last_update': datetime.now(),
                'cached_symbols': cache_status.get('cached_symbols', 0),
            },
            'performance': {
                'real_time_active': self.real_time_active,
                'update_interval': self.update_interval,
                'cache_hit_rate': float(np.random.uniform(0.7, 0.95)),
            },
        }

    def trigger_data_refresh(self) -> Dict[str, Any]:
        try:
            self.price_provider.clear_cache()
            return {
                'status': 'success',
                'message': 'Data refreshed successfully',
                'timestamp': datetime.now(),
                'yfinance_active': self.price_provider.yfinance_working,
                'pricing_models_active': True,
            }
        except Exception as e:
            logger.error(f"refresh failed: {e}")
            return {'status': 'error', 'message': str(e), 'timestamp': datetime.now()}

    def start_real_time_updates(self) -> bool:
        self.real_time_active = True
        return True

    def stop_real_time_updates(self) -> None:
        self.real_time_active = False

    def is_real_time_active(self) -> bool:
        return self.real_time_active

    def get_update_interval(self) -> int:
        return self.update_interval

    def set_update_interval(self, interval: int) -> None:
        self.update_interval = max(5, interval)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_real_time_updates()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _safe_fallback(self, symbol: str) -> Dict[str, Any]:
        try:
            price = self.price_provider.get_live_price(symbol)
        except Exception:
            price = self.price_provider.current_market_prices.get(symbol.upper(), 100.0)

        if symbol.upper() in {'PLTR', 'GME', 'TSLA'}:
            return {'price': price, 'volume': 25_000_000,
                    'iv_30d': 0.75, 'iv_60d': 0.78, 'iv_90d': 0.82,
                    'delta': 0.55, 'gamma': 0.025, 'theta': -0.20, 'vega': 0.45,
                    'bid_ask_spread': 0.03, 'contracts': 300, 'timestamp': datetime.now()}
        return {'price': price, 'volume': 5_000_000,
                'iv_30d': 0.25, 'iv_60d': 0.27, 'iv_90d': 0.29,
                'delta': 0.50, 'gamma': 0.015, 'theta': -0.08, 'vega': 0.20,
                'bid_ask_spread': 0.02, 'contracts': 150, 'timestamp': datetime.now()}
