"""
Dashboard data orchestrator.

This connector keeps the Streamlit UI honest about data provenance. Real or
delayed yfinance option chains are used when available; synthetic and fallback
surfaces are explicitly marked as such for the dashboard.
"""

from __future__ import annotations

import logging
import warnings
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
from src.pricing.implied_vol import ImpliedVolatilityCalculator
from src.quant.corporate_actions import CorporateActionProvider, expiry_corporate_action_metadata
from src.quant.dividends import DividendProvider, apply_dividends_to_options, expiry_dividend_metadata
from src.quant.rates import RiskFreeRateProvider, apply_curve_to_options, expiry_rate_metadata

logger = logging.getLogger(__name__)
OPTION_PRICE_SOURCES = {"midpoint", "mark", "last"}


class DashboardConnector:
    """Top-level data provider for the Streamlit dashboard."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.price_provider = RealTimePriceProvider()
        self.options_provider = YFinanceOptionsProvider(max_expirations=8)
        self.historical_loader = HistoricalPriceLoader()
        self.market_calendar = MarketCalendar()
        self.rate_provider = RiskFreeRateProvider()
        self.dividend_provider = DividendProvider()
        self.corporate_action_provider = CorporateActionProvider()
        self.options_generator = SyntheticOptionsGenerator(
            self.price_provider,
            rate_provider=self.rate_provider,
            dividend_provider=self.dividend_provider,
        )
        self.iv_calculator = ImpliedVolatilityCalculator()
        self.option_price_source = "mark"
        self.snapshot_dir = Path("data/snapshots")
        self.real_time_active = False
        self.update_interval = 30
        self.chain_cache_ttl = timedelta(minutes=5)
        self.chain_cache: Dict[str, Tuple[pd.DataFrame, Dict[str, Any], datetime]] = {}
        self.surface_metadata: Dict[str, Dict[str, Any]] = {}

    def configure_liquidity_filters(
        self,
        *,
        min_open_interest: Optional[int] = None,
        min_volume: Optional[int] = None,
        max_bid_ask_spread_pct: Optional[float] = None,
        max_quote_age_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Update option-chain liquidity filters and invalidate normalized caches."""
        configure = getattr(self.options_provider, "configure_liquidity_filters", None)
        if configure is None:
            return {}
        changed = configure(
            min_open_interest=min_open_interest,
            min_volume=min_volume,
            max_bid_ask_spread_pct=max_bid_ask_spread_pct,
            max_quote_age_days=max_quote_age_days,
        )
        if changed:
            self.chain_cache.clear()
            self.surface_metadata.clear()
        settings = getattr(self.options_provider, "liquidity_filter_settings", lambda: {})()
        return dict(settings)

    def configure_option_price_source(self, price_source: str) -> str:
        """Select which market price drives computed IV and surface fitting."""
        normalized = str(price_source or "mark").strip().lower()
        if normalized not in OPTION_PRICE_SOURCES:
            normalized = "mark"
        if normalized != self.option_price_source:
            self.option_price_source = normalized
            self.chain_cache.clear()
            self.surface_metadata.clear()
        return self.option_price_source

    # ------------------------------------------------------------------
    # Symbol-level data
    # ------------------------------------------------------------------

    def get_current_data(self, symbol: str) -> Dict[str, Any]:
        """Return current underlying and representative volatility data."""
        key = symbol.upper()
        now = datetime.now()
        try:
            spot = self.price_provider.get_live_price(key)
            rate_curve = self.rate_provider.get_curve()
            rate_30d = rate_curve.rate_for_dte(30).rate
            dividend_assumption = self.dividend_provider.get(key)
            corporate_actions = self.corporate_action_provider.get(key)
            dividend_30d = dividend_assumption.effective_yield(
                datetime.now() + timedelta(days=30),
                spot,
                rate_30d,
            )
            market_status = self.get_market_status()
            greeks = self.options_generator.calculate_greeks(
                key,
                spot,
                risk_free_rate=rate_30d,
                dividend_yield=dividend_30d,
            )
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
                "risk_free_rate_30d": chain_meta.get("risk_free_rate_30d") if chain_meta else rate_30d,
                "risk_free_rate_source": (
                    chain_meta.get("risk_free_rate_source") if chain_meta else rate_curve.source
                ),
                "dividend_yield_30d": chain_meta.get("effective_dividend_yield_30d") if chain_meta else dividend_30d,
                "dividend_source": chain_meta.get("dividend_source") if chain_meta else dividend_assumption.source,
                "corporate_action_warning_count": (
                    chain_meta.get("corporate_action_warning_count")
                    if chain_meta
                    else len(corporate_actions.warning_messages())
                ),
                "corporate_action_source": (
                    chain_meta.get("corporate_action_source") if chain_meta else corporate_actions.source
                ),
                "delta": greeks["delta"],
                "gamma": greeks["gamma"],
                "theta": greeks["theta"],
                "vega": greeks["vega"],
                "bid_ask_spread": chain_summary.get("median_spread_pct"),
                "contracts": chain_summary.get("contracts"),
                "liquidity_filtered_count": chain_meta.get("liquidity_filtered_count") if chain_meta else None,
                "rejection_reasons": chain_meta.get("rejection_reasons") if chain_meta else {},
                "option_price_source": (
                    chain_meta.get("option_price_source") if chain_meta else self.option_price_source
                ),
                "computed_iv_count": chain_meta.get("computed_iv_count") if chain_meta else None,
                "computed_iv_failed_count": chain_meta.get("computed_iv_failed_count") if chain_meta else None,
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

            surface_rate = meta.get("risk_free_rate_median") or meta.get("risk_free_rate_30d")
            surface_dividend = meta.get("effective_dividend_yield_median") or meta.get("effective_dividend_yield_30d")
            surface_chain = self._surface_iv_chain(chain)
            if surface_chain.empty:
                raise ValueError("No usable computed IV rows after selected price source")
            strikes, expiries, vols = build_surface(surface_chain, spot, key, risk_free_rate=surface_rate)
            self.surface_metadata[key] = {
                **meta,
                "surface_mode": meta.get("mode", "Live/Delayed"),
                "surface_source": meta.get("source", "yfinance"),
                "surface_points": int(np.size(vols)),
                "spot": spot,
                "spot_timestamp": datetime.now(),
                "surface_risk_free_rate": surface_rate,
                "surface_dividend_yield": surface_dividend,
                "surface_iv_input": "computedIV",
            }
            return strikes, expiries, vols
        except Exception as exc:
            logger.warning("Real chain surface failed for %s: %s", key, exc)
            chain = self.options_generator.create_chain(key)
            rate_curve = self.rate_provider.get_curve()
            rate_meta = self._rate_metadata(rate_curve, chain)
            dividend_assumption = self.dividend_provider.get(key)
            chain = apply_dividends_to_options(chain, dividend_assumption, spot)
            dividend_meta = self._dividend_metadata(dividend_assumption, chain, spot)
            corporate_actions = self.corporate_action_provider.get(key)
            corporate_meta = self._corporate_action_metadata(corporate_actions, chain)
            surface_rate = rate_meta.get("risk_free_rate_median") or rate_meta.get("risk_free_rate_30d")
            surface_dividend = (
                dividend_meta.get("effective_dividend_yield_median")
                or dividend_meta.get("effective_dividend_yield_30d")
            )
            strikes, expiries, vols = build_surface(chain, spot, key, risk_free_rate=surface_rate)
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
                "surface_risk_free_rate": surface_rate,
                "surface_dividend_yield": surface_dividend,
                "option_price_source": self.option_price_source,
                "surface_iv_input": "synthetic provider IV",
                **rate_meta,
                **dividend_meta,
                **corporate_meta,
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
                "liquidity_filters": getattr(self.options_provider, "liquidity_filter_settings", lambda: {})(),
                "option_price_source": self.option_price_source,
                "historical_cache_entries": len(self.historical_loader.cache),
                "risk_free_rate_source": self.rate_provider.get_curve().source,
                "risk_free_rate_mode": self.rate_provider.get_curve().mode,
                "dividend_cache_entries": len(self.dividend_provider._cache),
                "corporate_action_cache_entries": len(self.corporate_action_provider._cache),
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
                "rates_provider": self.rate_provider.get_curve().source,
                "dividends_provider": self.dividend_provider.preferred_source,
                "corporate_actions_provider": self.corporate_action_provider.preferred_source,
                "calendar_provider": market_status.get("market"),
                "data_delay_minutes": market_status.get("data_delay_minutes"),
                "liquidity_filters": getattr(self.options_provider, "liquidity_filter_settings", lambda: {})(),
                "option_price_source": self.option_price_source,
            },
        }

    def trigger_data_refresh(self) -> Dict[str, Any]:
        try:
            self.price_provider.clear_cache()
            self.chain_cache.clear()
            self.options_provider.clear_cache()
            self.historical_loader.clear_cache()
            self.rate_provider.clear_cache()
            self.dividend_provider.clear_cache()
            self.corporate_action_provider.clear_cache()
            self.surface_metadata.clear()
            return {
                "status": "success",
                "message": "Price, option-chain, rate, dividend, corporate-action, and surface caches cleared",
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
        rate_curve = self.rate_provider.get_curve()
        df = apply_curve_to_options(df, rate_curve)
        dividend_assumption = self.dividend_provider.get(symbol)
        df = apply_dividends_to_options(df, dividend_assumption, spot_price)
        corporate_actions = self.corporate_action_provider.get(symbol)
        df, price_meta = self._apply_option_price_source(df, spot_price)
        meta.update(self._rate_metadata(rate_curve, df))
        meta.update(self._dividend_metadata(dividend_assumption, df, spot_price))
        meta.update(price_meta)
        corporate_meta = self._corporate_action_metadata(corporate_actions, df)
        meta.update(corporate_meta)
        meta["warnings"] = self._merge_warnings(
            meta.get("warnings"),
            corporate_meta.get("corporate_action_warnings"),
        )
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
            "iv_source": "computed option chain" if "computedIV" in chain else "option chain",
        }
        iv_column = "computedIV" if "computedIV" in chain else "impliedVolatility"

        for target, name in ((30, "iv_30d"), (60, "iv_60d"), (90, "iv_90d")):
            sub = chain[np.abs(chain["daysToExpiration"] - target) <= 10].copy()
            if sub.empty:
                continue
            sub["atm_distance"] = np.abs(sub["strike"] - spot)
            atm = sub.sort_values("atm_distance").head(6)
            ivs = pd.to_numeric(atm[iv_column], errors="coerce").dropna()
            if not ivs.empty:
                out[name] = float(ivs.median())
        return out

    @staticmethod
    def _rate_metadata(rate_curve: Any, chain: pd.DataFrame) -> Dict[str, Any]:
        metadata = rate_curve.metadata_dict()
        metadata["expiry_rates"] = expiry_rate_metadata(chain, rate_curve)
        metadata["risk_free_rate_30d"] = rate_curve.rate_for_dte(30).rate
        if not chain.empty and "riskFreeRate" in chain.columns:
            rates = pd.to_numeric(chain["riskFreeRate"], errors="coerce").dropna()
            if not rates.empty:
                metadata["risk_free_rate_min"] = float(rates.min())
                metadata["risk_free_rate_max"] = float(rates.max())
                metadata["risk_free_rate_median"] = float(rates.median())
        return metadata

    @staticmethod
    def _dividend_metadata(assumption: Any, chain: pd.DataFrame, spot: float) -> Dict[str, Any]:
        metadata = assumption.metadata_dict()
        metadata["expiry_dividends"] = expiry_dividend_metadata(chain, assumption, spot)
        metadata["effective_dividend_yield_30d"] = assumption.effective_yield(
            datetime.now() + timedelta(days=30),
            spot,
        )
        if not chain.empty and "effectiveDividendYield" in chain.columns:
            yields = pd.to_numeric(chain["effectiveDividendYield"], errors="coerce").dropna()
            if not yields.empty:
                metadata["effective_dividend_yield_min"] = float(yields.min())
                metadata["effective_dividend_yield_max"] = float(yields.max())
                metadata["effective_dividend_yield_median"] = float(yields.median())
        return metadata

    @staticmethod
    def _corporate_action_metadata(action_snapshot: Any, chain: pd.DataFrame) -> Dict[str, Any]:
        metadata = action_snapshot.metadata_dict()
        if not chain.empty and "expiration" in chain.columns:
            metadata["expiry_corporate_actions"] = expiry_corporate_action_metadata(
                chain["expiration"],
                action_snapshot,
            )
        else:
            metadata["expiry_corporate_actions"] = {}
        return metadata

    @staticmethod
    def _merge_warnings(*groups: Any) -> List[str]:
        out: List[str] = []
        seen = set()
        for group in groups:
            for item in group or []:
                text = str(item)
                if text and text not in seen:
                    seen.add(text)
                    out.append(text)
        return out

    def _apply_option_price_source(self, chain: pd.DataFrame, spot: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute IV from the configured market-price source."""
        if chain.empty:
            return chain.copy(), {
                "option_price_source": self.option_price_source,
                "computed_iv_count": 0,
                "computed_iv_failed_count": 0,
            }

        df = chain.copy()
        selected_price, selected_source = self._selected_market_price(df, self.option_price_source)
        df["selectedMarketPrice"] = selected_price
        df["selectedPriceSource"] = selected_source
        df["ivInput"] = "computed"

        computed: List[float] = []
        failed = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _, row in df.iterrows():
                market_price = _float_or_nan(row.get("selectedMarketPrice"))
                strike = _float_or_nan(row.get("strike"))
                time_to_expiry = _float_or_nan(row.get("time_to_expiry"))
                if not np.isfinite(time_to_expiry):
                    days_to_expiry = _float_or_nan(row.get("daysToExpiration"))
                    time_to_expiry = days_to_expiry / 365.0 if np.isfinite(days_to_expiry) else np.nan
                rate = _float_or_nan(row.get("riskFreeRate"))
                dividend = _float_or_nan(row.get("effectiveDividendYield"))
                if not np.isfinite(dividend):
                    dividend = _float_or_nan(row.get("dividendYield"))
                option_type = str(row.get("type", "")).lower()

                if not all(np.isfinite(value) for value in (market_price, strike, time_to_expiry, rate)):
                    computed.append(np.nan)
                    failed += 1
                    continue

                iv, _ = self.iv_calculator.calculate_implied_vol(
                    market_price,
                    spot,
                    strike,
                    time_to_expiry,
                    rate,
                    option_type,
                    q=dividend if np.isfinite(dividend) else 0.0,
                    method="brent",
                )
                if iv is None or not np.isfinite(iv):
                    computed.append(np.nan)
                    failed += 1
                else:
                    computed.append(float(iv))

        df["computedIV"] = computed
        computed_count = int(pd.to_numeric(df["computedIV"], errors="coerce").notna().sum())
        return df, {
            "option_price_source": self.option_price_source,
            "computed_iv_count": computed_count,
            "computed_iv_failed_count": failed,
        }

    @staticmethod
    def _selected_market_price(chain: pd.DataFrame, price_source: str) -> Tuple[pd.Series, pd.Series]:
        """Return selected prices and row-level source labels."""
        mid = _numeric_series(chain, "mid")
        mark = _numeric_series(chain, "mark")
        last = _numeric_series(chain, "last")

        if price_source == "midpoint":
            return mid.where(mid > 0), pd.Series("midpoint", index=chain.index)
        if price_source == "last":
            return last.where(last > 0), pd.Series("last", index=chain.index)

        selected = mark.where(mark > 0)
        selected = selected.where(selected.notna(), mid.where(mid > 0))
        selected = selected.where(selected.notna(), last.where(last > 0))
        if "markSource" in chain:
            source = chain["markSource"].fillna("mark").astype(str)
        else:
            source = pd.Series("mark", index=chain.index)
        source = source.where(selected.notna(), "unavailable")
        source = source.where(~((mark.isna() | (mark <= 0)) & (mid > 0)), "midpoint")
        source = source.where(~((mark.isna() | (mark <= 0)) & (mid.isna() | (mid <= 0)) & (last > 0)), "last")
        return selected, source

    @staticmethod
    def _surface_iv_chain(chain: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with computed IV as the fitting input."""
        if chain.empty or "computedIV" not in chain:
            return pd.DataFrame()
        out = chain.copy()
        computed = pd.to_numeric(out["computedIV"], errors="coerce")
        out = out[computed.notna()].copy()
        out["impliedVolatility"] = pd.to_numeric(out["computedIV"], errors="coerce")
        return out

    def _safe_fallback(self, symbol: str) -> Dict[str, Any]:
        try:
            price = self.price_provider.get_live_price(symbol)
        except Exception:
            price = self.price_provider.current_market_prices.get(symbol.upper(), 100.0)

        rate_curve = self.rate_provider.get_curve()
        rate_30d = rate_curve.rate_for_dte(30).rate
        dividend_assumption = self.dividend_provider.get(symbol)
        corporate_actions = self.corporate_action_provider.get(symbol)
        dividend_30d = dividend_assumption.effective_yield(datetime.now() + timedelta(days=30), price, rate_30d)
        greeks = self.options_generator.calculate_greeks(
            symbol,
            price,
            risk_free_rate=rate_30d,
            dividend_yield=dividend_30d,
        )
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
            "risk_free_rate_30d": rate_30d,
            "risk_free_rate_source": rate_curve.source,
            "dividend_yield_30d": dividend_30d,
            "dividend_source": dividend_assumption.source,
            "corporate_action_warning_count": len(corporate_actions.warning_messages()),
            "corporate_action_source": corporate_actions.source,
            "delta": greeks["delta"],
            "gamma": greeks["gamma"],
            "theta": greeks["theta"],
            "vega": greeks["vega"],
            "bid_ask_spread": None,
            "contracts": None,
            "liquidity_filtered_count": None,
            "rejection_reasons": {},
            "option_price_source": self.option_price_source,
            "computed_iv_count": None,
            "computed_iv_failed_count": None,
            "market_status": market_status.get("session_state"),
            "market_reason": market_status.get("reason"),
            "data_delay_minutes": market_status.get("data_delay_minutes"),
            "timestamp": datetime.now(),
        }


def _float_or_nan(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")
