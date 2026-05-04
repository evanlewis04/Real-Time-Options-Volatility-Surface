"""
Dashboard data orchestrator.

This connector keeps the Streamlit UI honest about data provenance. Real or
delayed yfinance option chains are used when available; synthetic and fallback
surfaces are explicitly marked as such for the dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.surface_builder import build_surface
from src.data.historical import HistoricalPriceLoader
from src.data.market_calendar import MarketCalendar
from src.data.models import MarketDataSnapshot
from src.data.options_provider import OptionsChainMetadata, YFinanceOptionsProvider
from src.data.price_provider import RealTimePriceProvider
from src.data.snapshots import load_latest_snapshot, save_snapshot
from src.data.synthetic_options import SyntheticOptionsGenerator

logger = logging.getLogger(__name__)


class DashboardConnector:
    """Top-level data provider for the Streamlit dashboard."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.price_provider = RealTimePriceProvider()
        self.options_provider = YFinanceOptionsProvider(max_expirations=8)
        self.historical_loader = HistoricalPriceLoader()
        self.market_calendar = MarketCalendar()
        self.options_generator = SyntheticOptionsGenerator(self.price_provider)
        self.snapshot_dir = Path("data/snapshots")
        self.real_time_active = False
        self.update_interval = 30
        self.chain_cache_ttl = timedelta(minutes=5)
        self.chain_cache: Dict[str, Tuple[pd.DataFrame, Dict[str, Any], datetime]] = {}
        self.surface_metadata: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Symbol-level data
    # ------------------------------------------------------------------

    def get_current_data(self, symbol: str) -> Dict[str, Any]:
        """Return current underlying and representative volatility data."""
        key = symbol.upper()
        now = datetime.now()
        try:
            spot = self.price_provider.get_live_price(key)
            market_status = self.get_market_status()
            greeks = self.options_generator.calculate_greeks(key, spot)
            chain, chain_meta = self._cached_chain_if_present(key)
            chain_summary = self._summarize_chain(chain, spot) if chain is not None and not chain.empty else {}

            data_mode = "Live/Delayed" if self.price_provider.yfinance_working else "Fallback"
            source = "yfinance" if self.price_provider.yfinance_working else "simulated price fallback"
            if chain_meta:
                data_mode = chain_meta.get("mode", data_mode)

            return {
                "symbol": key,
                "price": spot,
                "price_source": source,
                "data_mode": data_mode,
                "iv_source": chain_summary.get("iv_source", "model profile"),
                "volume": chain_summary.get("volume"),
                "open_interest": chain_summary.get("open_interest"),
                "iv_30d": chain_summary.get("iv_30d", greeks["iv_30d"]),
                "iv_60d": chain_summary.get("iv_60d", greeks["iv_60d"]),
                "iv_90d": chain_summary.get("iv_90d", greeks["iv_90d"]),
                "delta": greeks["delta"],
                "gamma": greeks["gamma"],
                "theta": greeks["theta"],
                "vega": greeks["vega"],
                "bid_ask_spread": chain_summary.get("median_spread_pct"),
                "contracts": chain_summary.get("contracts"),
                "market_status": market_status.get("session_state"),
                "market_reason": market_status.get("reason"),
                "data_delay_minutes": market_status.get("data_delay_minutes"),
                "timestamp": now,
            }
        except Exception as exc:
            logger.error("get_current_data(%s) failed: %s", key, exc)
            return self._safe_fallback(key)

    def get_options_chain_snapshot(self, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Return normalized option-chain data and metadata for ``symbol``."""
        snapshot = self.get_market_data_snapshot(symbol)
        return snapshot.options_frame(), snapshot.metadata_dict()

    def get_market_data_snapshot(self, symbol: str) -> MarketDataSnapshot:
        """Return the canonical snapshot model for ``symbol``."""
        key = symbol.upper()
        spot_timestamp = datetime.now()
        spot = self.price_provider.get_live_price(key)
        chain, meta = self._get_options_chain(key, spot)
        if chain.empty:
            persisted = self.get_latest_persisted_snapshot(key)
            if persisted is not None:
                return replace(
                    persisted,
                    fallback_reason=meta.get("fallback_reason") or "Using latest persisted snapshot",
                    mode="Fallback",
                    cache_age=datetime.now() - persisted.spot_timestamp,
                )

        snapshot = MarketDataSnapshot.from_chain_frame(key, spot, spot_timestamp, chain, meta)
        if snapshot.options:
            try:
                save_snapshot(snapshot, self.snapshot_dir)
            except Exception as exc:
                logger.debug("Snapshot persistence failed for %s: %s", key, exc)
        return snapshot

    def get_latest_persisted_snapshot(self, symbol: str) -> Optional[MarketDataSnapshot]:
        """Load the latest local snapshot for replay/offline fallback."""
        return load_latest_snapshot(symbol.upper(), self.snapshot_dir)

    def get_market_status(self) -> Dict[str, Any]:
        """Return current market calendar state."""
        return self.market_calendar.status().as_dict()

    def get_vol_surface_data(self, symbol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return strikes, expiries in days, and IVs for the surface plot."""
        key = symbol.upper()
        spot = self.price_provider.get_live_price(key)
        try:
            chain, meta = self._get_options_chain(key, spot)
            if chain.empty:
                raise ValueError(meta.get("fallback_reason") or "No usable option-chain data")

            strikes, expiries, vols = build_surface(chain, spot, key)
            self.surface_metadata[key] = {
                **meta,
                "surface_mode": meta.get("mode", "Live/Delayed"),
                "surface_source": meta.get("source", "yfinance"),
                "surface_points": int(np.size(vols)),
                "spot": spot,
                "spot_timestamp": datetime.now(),
            }
            return strikes, expiries, vols
        except Exception as exc:
            logger.warning("Real chain surface failed for %s: %s", key, exc)
            chain = self.options_generator.create_chain(key)
            strikes, expiries, vols = build_surface(chain, spot, key)
            self.surface_metadata[key] = {
                "symbol": key,
                "source": "synthetic generator",
                "mode": "Synthetic",
                "surface_mode": "Synthetic",
                "surface_source": "Black-Scholes synthetic chain",
                "timestamp": datetime.now(),
                "raw_rows": len(chain),
                "valid_rows": len(chain),
                "rejected_rows": 0,
                "fallback_reason": str(exc),
                "warnings": ["Real option chain was unavailable; generated a synthetic chain."],
                "surface_points": int(np.size(vols)),
                "spot": spot,
                "spot_timestamp": datetime.now(),
            }
            return strikes, expiries, vols

    def get_surface_metadata(self, symbol: str) -> Dict[str, Any]:
        """Return latest surface metadata for ``symbol``."""
        return self.surface_metadata.get(symbol.upper(), {})

    # ------------------------------------------------------------------
    # Portfolio and cross-asset summaries
    # ------------------------------------------------------------------

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Return explicit empty-state portfolio metrics.

        The project does not yet have a real position book, so the dashboard
        should not show random VaR, Sharpe, or P&L as if they were live.
        """
        return {
            "configured": False,
            "message": "No position book configured",
            "total_value": None,
            "daily_pnl": None,
            "var_95": None,
            "sharpe_ratio": None,
            "max_drawdown": None,
            "volatility": None,
        }

    def get_correlation_matrix(self, symbols: Optional[Iterable[str]] = None, period: str = "6mo") -> pd.DataFrame:
        """Calculate realized-return correlations from historical closes."""
        symbols = [s.upper() for s in (symbols or []) if s]
        if len(symbols) < 2:
            return pd.DataFrame()

        returns = self.historical_loader.load_many_returns(symbols, period=period, min_points=20)
        if len(returns) < 2:
            return pd.DataFrame()
        frame = pd.concat(returns.values(), axis=1).dropna(how="all")
        return frame.corr()

    def get_historical_metrics(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Return historical realized-vol and return series for analytics panels."""
        key = symbol.upper()
        result = self.historical_loader.load(key, period)
        if not result.available:
            return {"available": False, "reason": result.fallback_reason or "No historical closes returned"}
        close = result.close()
        returns = result.returns()
        realized_20d = result.realized_vol(20)
        realized_60d = result.realized_vol(60)
        return {
            "available": True,
            "source": result.source,
            "close": close,
            "returns": returns,
            "realized_20d": realized_20d,
            "realized_60d": realized_60d,
            "last_close": float(close.iloc[-1]),
            "realized_20d_latest": float(realized_20d.dropna().iloc[-1]) if realized_20d.notna().any() else None,
            "realized_60d_latest": float(realized_60d.dropna().iloc[-1]) if realized_60d.notna().any() else None,
        }

    # ------------------------------------------------------------------
    # System / lifecycle
    # ------------------------------------------------------------------

    def get_system_health(self) -> Dict[str, Any]:
        cache_status = getattr(self.price_provider, "get_cache_status", lambda: {})()
        market_status = self.get_market_status()
        return {
            "overall": {
                "pricing_models_available": True,
                "yfinance_available": self.price_provider.yfinance_working,
                "black_scholes_active": True,
                "implied_vol_active": True,
                "vol_surface_active": True,
                "real_time_pricing": self.price_provider.yfinance_working,
                "last_update": datetime.now(),
                "cached_symbols": cache_status.get("cached_symbols", 0),
                "option_chain_cache_entries": len(self.chain_cache),
                "option_expiry_cache_entries": self.options_provider.cache_status().get("entries"),
                "historical_cache_entries": len(self.historical_loader.cache),
                "market_state": market_status.get("session_state"),
                "market_reason": market_status.get("reason"),
                "next_market_open": market_status.get("next_open"),
            },
            "performance": {
                "real_time_active": self.real_time_active,
                "update_interval": self.update_interval,
                "cache_hit_rate": None,
            },
            "data_contract": {
                "price_provider": "yfinance" if self.price_provider.yfinance_working else "simulated fallback",
                "options_provider": "yfinance delayed chains",
                "fallback_provider": "Black-Scholes synthetic chain",
                "calendar_provider": market_status.get("market"),
                "data_delay_minutes": market_status.get("data_delay_minutes"),
            },
        }

    def trigger_data_refresh(self) -> Dict[str, Any]:
        try:
            self.price_provider.clear_cache()
            self.chain_cache.clear()
            self.options_provider.clear_cache()
            self.historical_loader.clear_cache()
            self.surface_metadata.clear()
            return {
                "status": "success",
                "message": "Price, option-chain, and surface caches cleared",
                "timestamp": datetime.now(),
                "yfinance_active": self.price_provider.yfinance_working,
                "pricing_models_active": True,
            }
        except Exception as exc:
            logger.error("refresh failed: %s", exc)
            return {"status": "error", "message": str(exc), "timestamp": datetime.now()}

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_options_chain(self, symbol: str, spot: Optional[float] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        now = datetime.now()
        cached = self.chain_cache.get(symbol)
        if cached and now - cached[2] < self.chain_cache_ttl:
            df, meta, cached_at = cached
            meta = {**meta, "cache_age_seconds": int((now - cached_at).total_seconds())}
            return df.copy(), meta

        spot_price = spot if spot is not None else self.price_provider.get_live_price(symbol)
        df, meta_obj = self.options_provider.fetch_chain(symbol, spot_price)
        meta = meta_obj.as_dict() if isinstance(meta_obj, OptionsChainMetadata) else dict(meta_obj)
        meta["cache_age_seconds"] = 0
        self.chain_cache[symbol] = (df.copy(), meta, now)
        return df, meta

    def _cached_chain_if_present(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        cached = self.chain_cache.get(symbol)
        if not cached:
            return None, None
        df, meta, cached_at = cached
        if datetime.now() - cached_at >= self.chain_cache_ttl:
            return None, None
        return df.copy(), {**meta, "cache_age_seconds": int((datetime.now() - cached_at).total_seconds())}

    @staticmethod
    def _summarize_chain(chain: pd.DataFrame, spot: float) -> Dict[str, Any]:
        if chain.empty:
            return {}
        out: Dict[str, Any] = {
            "contracts": int(len(chain)),
            "volume": int(pd.to_numeric(chain.get("volume"), errors="coerce").fillna(0).sum()),
            "open_interest": int(pd.to_numeric(chain.get("openInterest"), errors="coerce").fillna(0).sum()),
            "median_spread_pct": float(chain["bidAskSpreadPct"].median()) if "bidAskSpreadPct" in chain else None,
            "iv_source": "option chain",
        }

        for target, name in ((30, "iv_30d"), (60, "iv_60d"), (90, "iv_90d")):
            sub = chain[np.abs(chain["daysToExpiration"] - target) <= 10].copy()
            if sub.empty:
                continue
            sub["atm_distance"] = np.abs(sub["strike"] - spot)
            atm = sub.sort_values("atm_distance").head(6)
            out[name] = float(atm["impliedVolatility"].median())
        return out

    def _safe_fallback(self, symbol: str) -> Dict[str, Any]:
        try:
            price = self.price_provider.get_live_price(symbol)
        except Exception:
            price = self.price_provider.current_market_prices.get(symbol.upper(), 100.0)

        greeks = self.options_generator.calculate_greeks(symbol, price)
        market_status = self.get_market_status()
        return {
            "symbol": symbol.upper(),
            "price": price,
            "price_source": "fallback",
            "data_mode": "Fallback",
            "iv_source": "model profile",
            "volume": None,
            "open_interest": None,
            "iv_30d": greeks["iv_30d"],
            "iv_60d": greeks["iv_60d"],
            "iv_90d": greeks["iv_90d"],
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "theta": greeks["theta"],
            "vega": greeks["vega"],
            "bid_ask_spread": None,
            "contracts": None,
            "market_status": market_status.get("session_state"),
            "market_reason": market_status.get("reason"),
            "data_delay_minutes": market_status.get("data_delay_minutes"),
            "timestamp": datetime.now(),
        }
