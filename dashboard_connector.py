"""
Dashboard data orchestrator.

This connector keeps the Streamlit UI honest about data provenance. Real or
delayed yfinance option chains are used when available; synthetic and fallback
surfaces are explicitly marked as such for the dashboard.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.surface_builder import build_surface
from src.config.settings import AppSettings, FitFilterSettings, load_app_settings
from src.data.demo_provider import DemoOptionsProvider
from src.data.historical import HistoricalPriceLoader
from src.data.market_calendar import MarketCalendar
from src.data.models import MarketDataSnapshot
from src.data.options_provider import OptionsChainMetadata, YFinanceOptionsProvider
from src.data.price_provider import RealTimePriceProvider
from src.data.snapshots import load_latest_snapshot, save_snapshot
from src.data.synthetic_options import SyntheticOptionsGenerator
from src.pricing.implied_vol import ImpliedVolatilityCalculator
from src.quant.american import american_pricing_metadata, apply_american_pricing
from src.quant.advanced_features import (
    broker_integration_abstraction,
    build_option_strategy,
    compare_saved_snapshots,
    create_async_refresh_engine,
    cross_sectional_vol_map,
    earnings_vol_event_engine,
    estimate_transaction_costs,
    evaluate_surface_alerts,
    export_analysis_notebook,
    forecast_volatility,
    generate_research_report,
    list_surface_workspaces,
    load_surface_workspace,
    ml_anomaly_detector,
    news_event_overlay,
    optimize_portfolio_hedges,
    paper_trading_simulator,
    parse_portfolio_positions,
    portfolio_risk_summary,
    relative_value_dashboard,
    run_signal_backtest,
    save_surface_workspace,
    strategy_scenario_engine,
    classify_vol_regime,
    watchlist_presets,
)
from src.quant.corporate_actions import CorporateActionProvider, expiry_corporate_action_metadata
from src.quant.dividends import DividendProvider, apply_dividends_to_options, expiry_dividend_metadata
from src.quant.arbitrage import apply_no_arbitrage_checks
from src.quant.events import EventCalendarProvider, MarketEvent, expiry_event_metadata
from src.quant.expected_move import expected_moves_by_expiry
from src.quant.forwards import apply_forward_metrics, expiry_forward_metadata
from src.quant.heston import calibrate_heston_research
from src.quant.iv_history import atm_iv_from_chain, iv_rank_percentile_from_snapshots
from src.quant.local_vol import dupire_local_vol_surface
from src.quant.model_selection import (
    MODEL_LABELS,
    apply_model_selection,
    contract_greeks_metadata,
    normalize_pricing_model,
    pricing_model_metadata,
)
from src.quant.price_decomposition import apply_price_decomposition, price_decomposition_metadata
from src.quant.quote_quality import apply_quote_reliability_scores
from src.quant.rates import RiskFreeRateProvider, apply_curve_to_options, expiry_rate_metadata
from src.quant.realized_vol import latest_realized_volatility, realized_volatility_estimators
from src.quant.sabr import calibrate_sabr_by_expiry
from src.quant.skew import delta_skew_by_expiry
from src.quant.smoothing import smoothing_summary
from src.quant.shocks import surface_shock_scenarios
from src.quant.surface_change import rich_cheap_scanner, surface_change_analytics
from src.quant.surface_prior import (
    blend_surface_with_prior,
    load_historical_surface_prior,
    surface_prior_comparison_records,
)
from src.quant.svi import (
    calibrate_ssvi_surface,
    calibrate_svi_by_expiry,
    fit_diagnostics_from_ssvi,
    fit_diagnostics_from_svi,
)
from src.utils.structured_logging import configure_structured_logging, log_event
from src.utils.timing import PerformanceRecorder

logger = logging.getLogger(__name__)
OPTION_PRICE_SOURCES = {"midpoint", "mark", "last"}


class DashboardConnector:
    """Top-level data provider for the Streamlit dashboard."""

    def __init__(self, config_file: Optional[str] = None, settings: Optional[AppSettings] = None):
        self.config_file = config_file
        self.settings = settings or load_app_settings()
        configure_structured_logging(self.settings.logging)
        self.performance = PerformanceRecorder()
        provider_settings = self.settings.providers
        self.price_provider = RealTimePriceProvider(cache_duration_seconds=provider_settings.price_cache_seconds)
        self.options_provider = YFinanceOptionsProvider(
            max_expirations=provider_settings.max_expirations,
            cache_ttl_seconds=provider_settings.chain_cache_seconds,
            max_quote_age_days=provider_settings.max_quote_age_days,
            min_open_interest=provider_settings.min_open_interest,
            min_volume=provider_settings.min_volume,
            max_bid_ask_spread_pct=provider_settings.max_bid_ask_spread_pct,
        )
        self.historical_loader = HistoricalPriceLoader()
        self.market_calendar = MarketCalendar()
        self.rate_provider = RiskFreeRateProvider()
        self.dividend_provider = DividendProvider()
        self.corporate_action_provider = CorporateActionProvider()
        self.event_provider = EventCalendarProvider()
        self.options_generator = SyntheticOptionsGenerator(
            self.price_provider,
            rate_provider=self.rate_provider,
            dividend_provider=self.dividend_provider,
        )
        self.demo_provider = DemoOptionsProvider(
            self.price_provider,
            rate_provider=self.rate_provider,
            dividend_provider=self.dividend_provider,
            random_seed=self.settings.demo.random_seed,
            max_expirations=self.settings.demo.max_expirations,
        )
        self.iv_calculator = ImpliedVolatilityCalculator()
        self.option_price_source = "mark"
        self.pricing_model = "bsm_dividends"
        self.fit_filters = self.settings.fit_filters
        self.snapshot_dir = Path(self.settings.dashboard.snapshot_dir)
        self.real_time_active = False
        self.update_interval = self.settings.dashboard.update_interval_seconds
        self.chain_cache_ttl = timedelta(seconds=provider_settings.chain_cache_seconds)
        self.chain_cache: Dict[str, Tuple[pd.DataFrame, Dict[str, Any], datetime]] = {}
        self.surface_metadata: Dict[str, Dict[str, Any]] = {}
        self.surface_grids: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.async_refresh_engine = create_async_refresh_engine(max_workers=2)

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
            self.surface_grids.clear()
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
            self.surface_grids.clear()
        return self.option_price_source

    def configure_pricing_model(self, pricing_model: str) -> str:
        """Select the model used for contract analytics and dashboard provenance."""
        normalized = normalize_pricing_model(pricing_model)
        if normalized != self.pricing_model:
            self.pricing_model = normalized
            self.chain_cache.clear()
            self.surface_metadata.clear()
            self.surface_grids.clear()
        return MODEL_LABELS[self.pricing_model]

    def configure_fit_filters(
        self,
        *,
        preset: Optional[str] = None,
        max_bid_ask_spread_pct: Optional[float] = None,
        max_quote_age_days: Optional[int] = None,
        min_volume: Optional[int] = None,
        min_open_interest: Optional[int] = None,
        moneyness_min: Optional[float] = None,
        moneyness_max: Optional[float] = None,
        max_raw_iv: Optional[float] = None,
        no_arbitrage_policy: Optional[str] = None,
        last_only_policy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update surface-fit filters independently from chain display filters."""
        updates = {
            "preset": preset,
            "max_bid_ask_spread_pct": max_bid_ask_spread_pct,
            "max_quote_age_days": max_quote_age_days,
            "min_volume": min_volume,
            "min_open_interest": min_open_interest,
            "moneyness_min": moneyness_min,
            "moneyness_max": moneyness_max,
            "max_raw_iv": max_raw_iv,
            "no_arbitrage_policy": no_arbitrage_policy,
            "last_only_policy": last_only_policy,
        }
        clean_updates = {key: value for key, value in updates.items() if value is not None}
        next_filters = replace(self.fit_filters, **clean_updates)
        if next_filters != self.fit_filters:
            self.fit_filters = next_filters
            self.chain_cache.clear()
            self.surface_metadata.clear()
            self.surface_grids.clear()
        return asdict(self.fit_filters)

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
                "event_count": chain_meta.get("event_count") if chain_meta else None,
                "event_source": chain_meta.get("event_source") if chain_meta else None,
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
                "pricing_model": chain_meta.get("pricing_model") if chain_meta else self.pricing_model,
                "pricing_model_label": (
                    chain_meta.get("pricing_model_label")
                    if chain_meta
                    else pricing_model_metadata(self.pricing_model)["pricing_model_label"]
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
            with self.performance.measure(
                "surface_build",
                symbol=key,
                provider=meta.get("source"),
                source=meta.get("mode"),
                cache_hit=bool(meta.get("cache_age_seconds")),
            ):
                strikes, expiries, vols = build_surface(surface_chain, spot, key, risk_free_rate=surface_rate)
            metadata = {
                **meta,
                "surface_mode": meta.get("mode", "Live/Delayed"),
                "surface_source": meta.get("source", "yfinance"),
                "surface_points": int(np.size(vols)),
                "spot": spot,
                "spot_timestamp": datetime.now(),
                "surface_risk_free_rate": surface_rate,
                "surface_dividend_yield": surface_dividend,
                "surface_iv_input": "computedIV",
                "surface_smoothing": smoothing_summary(strikes, expiries, vols),
                **self._svi_metadata(surface_chain, spot),
                **self._heston_metadata(surface_chain, spot),
                **self._sabr_metadata(key, surface_chain, spot),
            }
            metadata.update(self._surface_quality_metadata(surface_chain, metadata))
            prior = load_historical_surface_prior(
                key,
                self.snapshot_dir,
                as_of=metadata.get("timestamp") or metadata.get("spot_timestamp"),
            )
            current_smoothing = metadata.get("surface_smoothing")
            prior_comparison = surface_prior_comparison_records(strikes, expiries, vols, spot, prior)
            vols, prior_blend = blend_surface_with_prior(
                strikes,
                expiries,
                vols,
                spot,
                prior,
                quality_score=metadata.get("surface_quality_score"),
            )
            metadata.update(
                {
                    "historical_surface_prior": prior.metadata(),
                    "historical_surface_prior_grid": prior.records(),
                    "surface_prior_comparison": prior_comparison,
                    "surface_prior_comparison_available": bool(prior_comparison),
                    "surface_prior": prior_blend,
                    "surface_prior_applied": bool(prior_blend.get("applied")),
                    "surface_prior_source": prior_blend.get("prior_source"),
                    "surface_prior_age_days": prior_blend.get("prior_age_days"),
                    "surface_prior_blend_weight": prior_blend.get("blend_weight"),
                    "surface_prior_overlap_count": prior_blend.get("overlap_count"),
                    "surface_estimate_type": (
                        "prior_assisted_estimate" if prior_blend.get("applied") else "current_fit_estimate"
                    ),
                    "current_surface_smoothing": current_smoothing,
                    "surface_smoothing": smoothing_summary(strikes, expiries, vols),
                }
            )
            metadata.update(self._local_vol_metadata(strikes, expiries, vols, spot, metadata))
            metadata.update(self._iv_history_metadata(key, surface_chain, spot))
            metadata.update(self._surface_change_metadata(key, surface_chain, spot, metadata))
            metadata.update(self._rich_cheap_metadata(surface_chain, metadata))
            metadata.update(self._surface_shock_metadata(surface_chain, spot))
            self.surface_metadata[key] = metadata
            self.surface_grids[key] = (strikes, expiries, vols)
            return strikes, expiries, vols
        except Exception as exc:
            logger.warning("Real chain surface failed for %s: %s", key, exc)
            chain, demo_meta_obj = self.demo_provider.fetch_chain(key, spot, fallback_reason=str(exc))
            demo_meta = demo_meta_obj.as_dict()
            rate_curve = self.rate_provider.get_curve()
            chain = apply_curve_to_options(chain, rate_curve)
            rate_meta = self._rate_metadata(rate_curve, chain)
            dividend_assumption = self.dividend_provider.get(key)
            chain = apply_dividends_to_options(chain, dividend_assumption, spot)
            chain = apply_forward_metrics(chain, spot)
            dividend_meta = self._dividend_metadata(dividend_assumption, chain, spot)
            forward_meta = self._forward_metadata(chain)
            corporate_actions = self.corporate_action_provider.get(key)
            event_snapshot = self.event_provider.get(key)
            corporate_meta = self._corporate_action_metadata(corporate_actions, chain)
            event_meta = self._event_metadata(event_snapshot, chain, dividend_assumption, corporate_actions)
            chain, arbitrage_meta = apply_no_arbitrage_checks(chain, spot, price_column="last")
            chain = apply_price_decomposition(chain, spot)
            chain = apply_american_pricing(chain, spot)
            chain = apply_model_selection(chain, spot, self.pricing_model)
            surface_rate = rate_meta.get("risk_free_rate_median") or rate_meta.get("risk_free_rate_30d")
            surface_dividend = (
                dividend_meta.get("effective_dividend_yield_median")
                or dividend_meta.get("effective_dividend_yield_30d")
            )
            with self.performance.measure(
                "surface_build",
                symbol=key,
                provider=demo_meta.get("source"),
                source=demo_meta.get("mode"),
                fallback_reason=str(exc),
            ):
                strikes, expiries, vols = build_surface(chain, spot, key, risk_free_rate=surface_rate)
            metadata = {
                **demo_meta,
                "symbol": key,
                "source": demo_meta.get("source", "demo synthetic provider"),
                "mode": "Synthetic",
                "surface_mode": "Synthetic",
                "surface_source": demo_meta.get("source", "demo synthetic provider"),
                "timestamp": datetime.now(),
                "raw_rows": len(chain),
                "valid_rows": len(chain),
                "rejected_rows": 0,
                "fallback_reason": str(exc),
                "warnings": self._merge_warnings(
                    demo_meta.get("warnings"),
                    ["Real option chain was unavailable; generated a deterministic demo chain."],
                ),
                "surface_points": int(np.size(vols)),
                "spot": spot,
                "spot_timestamp": datetime.now(),
                "surface_risk_free_rate": surface_rate,
                "surface_dividend_yield": surface_dividend,
                "option_price_source": self.option_price_source,
                **pricing_model_metadata(self.pricing_model),
                "surface_iv_input": "synthetic provider IV",
                "surface_smoothing": smoothing_summary(strikes, expiries, vols),
                **self._svi_metadata(chain, spot, iv_column="impliedVolatility"),
                **self._heston_metadata(chain, spot, iv_column="impliedVolatility"),
                **self._sabr_metadata(key, chain, spot, iv_column="impliedVolatility"),
                **self._expected_move_metadata(chain, spot, iv_column="impliedVolatility", price_column="last"),
                **rate_meta,
                **dividend_meta,
                **forward_meta,
                **corporate_meta,
                **event_meta,
                **arbitrage_meta,
                **price_decomposition_metadata(chain),
                **american_pricing_metadata(chain),
                **contract_greeks_metadata(chain, self.pricing_model),
            }
            metadata.update(self._surface_quality_metadata(chain, metadata))
            metadata.update(self._local_vol_metadata(strikes, expiries, vols, spot, metadata))
            metadata.update(self._iv_history_metadata(key, chain, spot, iv_column="impliedVolatility"))
            metadata.update(
                self._surface_change_metadata(key, chain, spot, metadata, iv_column="impliedVolatility")
            )
            metadata.update(self._rich_cheap_metadata(chain, metadata, iv_column="impliedVolatility"))
            metadata.update(self._surface_shock_metadata(chain, spot, iv_column="impliedVolatility"))
            self.surface_metadata[key] = metadata
            self.surface_grids[key] = (strikes, expiries, vols)
            return strikes, expiries, vols

    def get_surface_metadata(self, symbol: str) -> Dict[str, Any]:
        """Return latest surface metadata for ``symbol``."""
        return self.surface_metadata.get(symbol.upper(), {})

    # ------------------------------------------------------------------
    # Portfolio and cross-asset summaries
    # ------------------------------------------------------------------

    def get_portfolio_metrics(self, position_csv: Any | None = None) -> Dict[str, Any]:
        """Price uploaded option positions and aggregate portfolio risk."""
        if position_csv is None:
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

        parsed = parse_portfolio_positions(position_csv)
        if not parsed.get("available"):
            return {
                "configured": True,
                "available": False,
                "reason": parsed.get("reason"),
                "parse_errors": parsed.get("errors") or [],
                "positions": [],
            }

        market_data: Dict[str, Dict[str, Any]] = {}
        grids: Dict[str, Tuple[Any, Any, Any]] = {}
        for symbol in sorted({row["symbol"] for row in parsed.get("positions") or []}):
            snapshot = self.get_market_data_snapshot(symbol)
            market_data[symbol] = {"spot": snapshot.spot, "chain": snapshot.options_frame()}
            grids[symbol] = self.surface_grids.get(symbol, (None, None, None))
            if grids[symbol][0] is None:
                try:
                    grids[symbol] = self.get_vol_surface_data(symbol)
                except Exception:
                    logger.debug("Portfolio surface grid unavailable for %s", symbol)

        summary = portfolio_risk_summary(parsed.get("positions") or [], market_data, surface_grids=grids)
        summary["parse_errors"] = parsed.get("errors") or []
        return summary

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
        realized_estimators = realized_volatility_estimators(result.frame, windows=(20, 60))
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
            "realized_estimators": realized_estimators,
            "realized_estimator_latest": latest_realized_volatility(realized_estimators),
        }

    def get_relative_value_dashboard(self, left_symbol: str, right_symbol: str) -> Dict[str, Any]:
        """Return a pair comparison view with normalized volatility overlays."""
        left = self._advanced_symbol_profile(left_symbol)
        right = self._advanced_symbol_profile(right_symbol)
        return relative_value_dashboard(left, right)

    def get_cross_sectional_vol_map(self, symbols: Iterable[str]) -> Dict[str, Any]:
        """Return sorted cross-sectional volatility opportunities for a universe."""
        profiles = [self._advanced_symbol_profile(symbol) for symbol in symbols if symbol]
        return cross_sectional_vol_map(profiles)

    def get_earnings_event_engine(self, symbol: str) -> Dict[str, Any]:
        """Return an implied earnings move card for ``symbol`` when events are available."""
        key = symbol.upper()
        snapshot = self.get_market_data_snapshot(key)
        events = (snapshot.metadata_dict().get("events") or [])
        return earnings_vol_event_engine(key, snapshot.options_frame(), snapshot.spot, events)

    def get_strategy_analytics(self, symbol: str, strategy_type: str) -> Dict[str, Any]:
        """Build and price a template strategy against the current fitted surface."""
        key = symbol.upper()
        snapshot = self.get_market_data_snapshot(key)
        strikes, expiries, vols = self.surface_grids.get(key, (None, None, None))
        if strikes is None or expiries is None or vols is None:
            strikes, expiries, vols = self.get_vol_surface_data(key)
        return build_option_strategy(
            snapshot.options_frame(),
            snapshot.spot,
            strategy_type,
            strike_grid=strikes,
            expiry_grid=expiries,
            surface=vols,
        )

    def get_strategy_scenarios(
        self,
        symbol: str,
        strategy_type: str,
        *,
        spot_shifts: list[float] | None = None,
        time_pass_days: list[float] | None = None,
        vol_shifts: list[float] | None = None,
        skew_shifts: list[float] | None = None,
    ) -> Dict[str, Any]:
        """Return strategy P&L scenario grids for the selected template."""
        strategy = self.get_strategy_analytics(symbol, strategy_type)
        spot = self.get_market_data_snapshot(symbol).spot
        return strategy_scenario_engine(
            strategy,
            spot,
            spot_shifts=spot_shifts,
            time_pass_days=time_pass_days,
            vol_shifts=vol_shifts,
            skew_shifts=skew_shifts,
        )

    def get_portfolio_optimization(
        self,
        position_csv: Any | None,
        objective: str = "delta-neutral",
        *,
        theta_target: float = 0.0,
    ) -> Dict[str, Any]:
        """Return hedge suggestions for an uploaded portfolio."""
        portfolio = self.get_portfolio_metrics(position_csv)
        return optimize_portfolio_hedges(portfolio, objective=objective, theta_target=theta_target)

    def get_surface_alerts(
        self,
        symbol: str,
        *,
        config: Dict[str, Any] | None = None,
        log_path: str | Path | None = "data/alerts/surface_alerts.jsonl",
    ) -> Dict[str, Any]:
        """Evaluate and locally log configured surface alerts."""
        key = symbol.upper()
        if key not in self.surface_metadata:
            try:
                self.get_vol_surface_data(key)
            except Exception:
                logger.debug("Surface metadata unavailable before alert evaluation for %s", key)
        return evaluate_surface_alerts(
            key,
            self.surface_metadata.get(key, {}),
            self.get_current_data(key),
            config=config,
            log_path=log_path,
        )

    def get_watchlist_presets(self) -> Dict[str, List[str]]:
        """Return predefined watchlist universes."""
        events = []
        for symbol in ("AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA", "SPY", "QQQ"):
            try:
                events.extend((self.event_provider.get(symbol).metadata_dict()).get("events") or [])
            except Exception:
                logger.debug("Watchlist event lookup failed for %s", symbol)
        return watchlist_presets(events)

    def save_workspace(
        self,
        workspace: Dict[str, Any],
        directory: str | Path = "data/workspaces",
        *,
        name: str | None = None,
    ) -> Dict[str, Any]:
        """Persist a reloadable local dashboard workspace."""
        return save_surface_workspace(workspace, directory, name=name)

    def load_workspace(self, path: str | Path) -> Dict[str, Any]:
        """Load a previously saved local dashboard workspace."""
        return load_surface_workspace(path)

    def list_workspaces(self, directory: str | Path = "data/workspaces") -> List[Dict[str, Any]]:
        """List saved local workspace configs."""
        return list_surface_workspaces(directory)

    def compare_saved_snapshots(self, left: Any, right: Any) -> Dict[str, Any]:
        """Compare two persisted or in-memory market snapshots."""
        return compare_saved_snapshots(left, right)

    def estimate_transaction_costs(self, trades: Any, **kwargs: Any) -> Dict[str, Any]:
        """Estimate explicit spread, slippage, commission, assignment, and exercise costs."""
        return estimate_transaction_costs(trades, **kwargs)

    def run_signal_backtest(self, observations: Any, **kwargs: Any) -> Dict[str, Any]:
        """Run a deterministic offline signal backtest."""
        return run_signal_backtest(observations, **kwargs)

    def run_paper_trading_simulator(self, orders: Any, marks: Any, **kwargs: Any) -> Dict[str, Any]:
        """Track local paper-trading orders and marks without broker connectivity."""
        return paper_trading_simulator(orders, marks, **kwargs)

    def get_broker_interface(self, positions: Any | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Return the read-only broker abstraction with live trading disabled."""
        return broker_integration_abstraction(positions, **kwargs)

    def export_analysis_notebook(self, analysis: Dict[str, Any], path: str | Path, **kwargs: Any) -> Dict[str, Any]:
        """Export the supplied analysis payload to a reproducible local notebook."""
        return export_analysis_notebook(analysis, path, **kwargs)

    def generate_research_report(
        self,
        symbol: str,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a local HTML research report for the current dashboard state."""
        key = symbol.upper()
        if key not in self.surface_metadata:
            try:
                self.get_vol_surface_data(key)
            except Exception:
                logger.debug("Surface metadata unavailable before report generation for %s", key)
        current = self.get_current_data(key)
        metadata = self.surface_metadata.get(key, {})
        report_path = Path(path) if path is not None else Path("reports") / f"{key}_surface_report.html"
        analysis = {
            "symbol": key,
            "spot": current.get("price"),
            "data_timestamp": (metadata.get("timestamp") or current.get("timestamp") or datetime.now()).isoformat()
            if hasattr(metadata.get("timestamp") or current.get("timestamp") or datetime.now(), "isoformat")
            else str(metadata.get("timestamp") or current.get("timestamp") or datetime.now()),
            "model_assumptions": metadata.get("pricing_model_label") or current.get("pricing_model_label"),
            "surface_summary": {
                "atm_iv": current.get("iv_30d"),
                "iv_rank": metadata.get("iv_rank"),
                "iv_percentile": metadata.get("iv_percentile"),
                "surface_points": metadata.get("surface_points"),
                "term_slope": (metadata.get("surface_smoothing") or {}).get("term_slope"),
            },
            "diagnostics": {
                "surface_quality_score": metadata.get("surface_quality_score"),
                "fit_diagnostics": metadata.get("fit_diagnostics"),
                "warnings": metadata.get("warnings"),
                "source": metadata.get("surface_source") or current.get("price_source"),
                "mode": metadata.get("surface_mode") or current.get("data_mode"),
            },
            "provenance": {
                "surface_source": metadata.get("surface_source"),
                "surface_mode": metadata.get("surface_mode"),
                "option_price_source": metadata.get("option_price_source"),
            },
        }
        return generate_research_report(analysis, report_path, title=f"{key} Volatility Surface Report", **kwargs)

    def get_ml_anomaly_detector(self, symbol: str, observations: Any | None = None) -> Dict[str, Any]:
        """Detect anomalous local surface moves or residuals for ``symbol``."""
        return ml_anomaly_detector(observations or self._local_snapshot_features(symbol))

    def get_vol_regime_classifier(self, symbol: str, observations: Any | None = None) -> Dict[str, Any]:
        """Classify the selected symbol's volatility regime with historical analogs."""
        key = symbol.upper()
        current = self._advanced_symbol_profile(key)
        return classify_vol_regime(observations or self._local_snapshot_features(key), current=current)

    def get_forecasting_module(self, symbol: str, observations: Any | None = None) -> Dict[str, Any]:
        """Run deterministic volatility forecasting baselines for ``symbol``."""
        return forecast_volatility(observations or self._local_snapshot_features(symbol))

    def get_news_event_overlay(self, symbol: str, surface_jumps: Any | None = None) -> Dict[str, Any]:
        """Return trusted event overlay markers for the current symbol."""
        key = symbol.upper()
        events = self.event_provider.get(key).metadata_dict().get("events") or []
        if surface_jumps is None:
            surface_jumps = (self.surface_metadata.get(key, {}).get("surface_change_heatmaps") or {}).get("records") or []
        return news_event_overlay(events, surface_jumps)

    def request_async_refresh(self, symbol: str) -> Dict[str, Any]:
        """Schedule a nonblocking refresh of current data, chain, and surface for ``symbol``."""
        key = symbol.upper()

        def loader() -> Dict[str, Any]:
            current = self.get_current_data(key)
            snapshot = self.get_market_data_snapshot(key)
            self.get_vol_surface_data(key)
            return {
                "symbol": key,
                "price": current.get("price"),
                "option_rows": len(snapshot.options),
                "refreshed_at": datetime.now().isoformat(),
            }

        return self.async_refresh_engine.request_refresh(key, loader)

    def get_async_refresh_status(self, symbol: str | None = None) -> Dict[str, Any]:
        """Return pending/completed async refresh state without blocking the dashboard."""
        return self.async_refresh_engine.snapshot(symbol.upper() if symbol else None)

    def _advanced_symbol_profile(self, symbol: str) -> Dict[str, Any]:
        key = symbol.upper()
        current = self.get_current_data(key)
        metadata = self.surface_metadata.get(key, {})
        historical = {"available": False}
        realized_latest = {}
        if metadata:
            try:
                historical = self.get_historical_metrics(key)
                realized_latest = historical.get("realized_estimator_latest") or {}
            except Exception:
                logger.debug("Historical profile metrics unavailable for %s", key)
        return {
            "symbol": key,
            "iv_30d": current.get("iv_30d"),
            "iv_60d": current.get("iv_60d"),
            "iv_90d": current.get("iv_90d"),
            "iv_rank": metadata.get("iv_rank"),
            "iv_percentile": metadata.get("iv_percentile"),
            "front_risk_reversal_25d": metadata.get("front_risk_reversal_25d"),
            "term_slope": (metadata.get("surface_smoothing") or {}).get("term_slope"),
            "realized_20d_latest": historical.get("realized_20d_latest")
            or realized_latest.get("close_to_close_20d"),
            "mode": current.get("data_mode"),
            "source": current.get("iv_source"),
        }

    def _local_snapshot_features(self, symbol: str) -> List[Dict[str, Any]]:
        key = symbol.upper()
        observations: List[Dict[str, Any]] = []
        snapshot = self.get_latest_persisted_snapshot(key)
        metadata = self.surface_metadata.get(key, {})
        if snapshot is not None:
            frame = snapshot.options_frame()
            computed_iv = (
                pd.to_numeric(frame["computedIV"], errors="coerce")
                if not frame.empty and "computedIV" in frame
                else pd.Series(dtype=float)
            )
            observations.append(
                {
                    "symbol": key,
                    "timestamp": snapshot.spot_timestamp,
                    "atm_iv": float(computed_iv.median()) if computed_iv.notna().any() else None,
                    "realized_vol": metadata.get("realized_20d_latest"),
                    "skew_25d": metadata.get("front_risk_reversal_25d"),
                    "term_slope": (metadata.get("surface_smoothing") or {}).get("term_slope"),
                    "fit_rmse": (metadata.get("fit_diagnostics") or {}).get("rmse"),
                    "data_quality_score": metadata.get("surface_quality_score"),
                }
            )
        if metadata:
            observations.append(
                {
                    "symbol": key,
                    "timestamp": metadata.get("timestamp") or datetime.now(),
                    "atm_iv": metadata.get("front_atm_iv") or metadata.get("atm_iv"),
                    "iv_change": metadata.get("atm_iv_change"),
                    "realized_vol": metadata.get("realized_20d_latest"),
                    "skew_25d": metadata.get("front_risk_reversal_25d"),
                    "term_slope": (metadata.get("surface_smoothing") or {}).get("term_slope"),
                    "fit_rmse": (metadata.get("fit_diagnostics") or {}).get("rmse"),
                    "data_quality_score": metadata.get("surface_quality_score"),
                }
            )
        return observations

    # ------------------------------------------------------------------
    # System / lifecycle
    # ------------------------------------------------------------------

    def get_system_health(self) -> Dict[str, Any]:
        cache_status = getattr(self.price_provider, "get_cache_status", lambda: {})()
        market_status = self.get_market_status()
        option_cache_status = getattr(self.options_provider, "cache_status", lambda: {})()
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
                "option_expiry_cache_entries": option_cache_status.get("entries"),
                "liquidity_filters": getattr(self.options_provider, "liquidity_filter_settings", lambda: {})(),
                "option_price_source": self.option_price_source,
                "pricing_model": MODEL_LABELS[self.pricing_model],
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
                "slowest_steps": self.performance.slowest(8),
                "recent_steps": self.performance.recent(20),
            },
            "data_contract": {
                "price_provider": "yfinance" if self.price_provider.yfinance_working else "simulated fallback",
                "options_provider": "yfinance delayed chains",
                "fallback_provider": self.demo_provider.source,
                "rates_provider": self.rate_provider.get_curve().source,
                "dividends_provider": self.dividend_provider.preferred_source,
                "corporate_actions_provider": self.corporate_action_provider.preferred_source,
                "calendar_provider": market_status.get("market"),
                "data_delay_minutes": market_status.get("data_delay_minutes"),
                "liquidity_filters": getattr(self.options_provider, "liquidity_filter_settings", lambda: {})(),
                "option_price_source": self.option_price_source,
                "pricing_model": MODEL_LABELS[self.pricing_model],
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
            self.event_provider.clear_cache()
            self.surface_metadata.clear()
            self.surface_grids.clear()
            return {
                "status": "success",
                "message": "Price, option-chain, rate, dividend, corporate-action, event, and surface caches cleared",
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
            self.performance.record(
                "options_chain_fetch",
                0.0,
                symbol=symbol,
                provider=meta.get("source"),
                source=meta.get("mode"),
                cache_hit=True,
                fallback_reason=meta.get("fallback_reason"),
            )
            log_event(
                logger,
                "provider_fetch",
                symbol=symbol,
                provider=meta.get("source"),
                source=meta.get("mode"),
                latency_ms=0.0,
                cache_hit=True,
                fallback_reason=meta.get("fallback_reason"),
            )
            return df.copy(), meta

        spot_price = spot if spot is not None else self.price_provider.get_live_price(symbol)
        with self.performance.measure(
            "options_chain_fetch",
            symbol=symbol,
            provider=self.options_provider.__class__.__name__,
            cache_hit=False,
        ):
            df, meta_obj = self.options_provider.fetch_chain(symbol, spot_price)
        meta = meta_obj.as_dict() if isinstance(meta_obj, OptionsChainMetadata) else dict(meta_obj)
        latest_timing = self.performance.records[-1] if self.performance.records else None
        if latest_timing is not None:
            meta["provider_latency_ms"] = latest_timing.latency_ms
        log_event(
            logger,
            "provider_fetch",
            symbol=symbol,
            provider=meta.get("source") or self.options_provider.__class__.__name__,
            source=meta.get("mode"),
            latency_ms=meta.get("provider_latency_ms"),
            cache_hit=False,
            fallback_reason=meta.get("fallback_reason"),
        )
        rate_curve = self.rate_provider.get_curve()
        df = apply_curve_to_options(df, rate_curve)
        dividend_assumption = self.dividend_provider.get(symbol)
        df = apply_dividends_to_options(df, dividend_assumption, spot_price)
        df = apply_forward_metrics(df, spot_price)
        corporate_actions = self.corporate_action_provider.get(symbol)
        event_snapshot = self.event_provider.get(symbol)
        df, price_meta = self._apply_option_price_source(df, spot_price)
        df = apply_price_decomposition(df, spot_price)
        df = apply_american_pricing(df, spot_price)
        df = apply_model_selection(df, spot_price, self.pricing_model)
        df, parity_meta = self._apply_parity_checks(df, spot_price)
        df, arbitrage_meta = apply_no_arbitrage_checks(df, spot_price)
        meta.update(self._rate_metadata(rate_curve, df))
        meta.update(self._dividend_metadata(dividend_assumption, df, spot_price))
        meta.update(self._forward_metadata(df))
        meta.update(price_meta)
        meta.update(price_decomposition_metadata(df))
        meta.update(american_pricing_metadata(df))
        meta.update(contract_greeks_metadata(df, self.pricing_model))
        meta.update(parity_meta)
        meta.update(arbitrage_meta)
        meta.update(self._fit_filter_metadata())
        df, reliability_meta = apply_quote_reliability_scores(df, meta, self.fit_filters)
        meta.update(reliability_meta)
        meta.update(self._skew_metadata(df, spot_price))
        meta.update(self._expected_move_metadata(df, spot_price))
        meta.update(self._data_quality_metadata(df, meta))
        corporate_meta = self._corporate_action_metadata(corporate_actions, df)
        meta.update(corporate_meta)
        event_meta = self._event_metadata(event_snapshot, df, dividend_assumption, corporate_actions)
        meta.update(event_meta)
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
    def _forward_metadata(chain: pd.DataFrame) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"expiry_forwards": expiry_forward_metadata(chain)}
        if not chain.empty and "forwardPrice" in chain.columns:
            forwards = pd.to_numeric(chain["forwardPrice"], errors="coerce").dropna()
            if not forwards.empty:
                metadata["forward_price_min"] = float(forwards.min())
                metadata["forward_price_max"] = float(forwards.max())
                metadata["forward_price_median"] = float(forwards.median())
        if not chain.empty and "discountFactor" in chain.columns:
            discounts = pd.to_numeric(chain["discountFactor"], errors="coerce").dropna()
            if not discounts.empty:
                metadata["discount_factor_min"] = float(discounts.min())
                metadata["discount_factor_max"] = float(discounts.max())
                metadata["discount_factor_median"] = float(discounts.median())
        return metadata

    @staticmethod
    def _skew_metadata(chain: pd.DataFrame, spot: float) -> Dict[str, Any]:
        skew = delta_skew_by_expiry(chain, spot)
        if skew.empty:
            return {"delta_skew": [], "front_risk_reversal_25d": None, "front_butterfly_25d": None}

        records = skew.replace({np.nan: None}).to_dict("records")
        front = records[0]
        return {
            "delta_skew": records,
            "front_risk_reversal_25d": front.get("risk_reversal_25d"),
            "front_butterfly_25d": front.get("butterfly_25d"),
        }

    @staticmethod
    def _expected_move_metadata(
        chain: pd.DataFrame,
        spot: float,
        iv_column: str = "computedIV",
        price_column: str = "selectedMarketPrice",
    ) -> Dict[str, Any]:
        moves = expected_moves_by_expiry(chain, spot, iv_column=iv_column, price_column=price_column)
        if moves.empty:
            return {
                "expected_moves": [],
                "front_expected_move": None,
                "front_expected_move_pct": None,
                "front_expected_move_method": None,
            }

        records = moves.replace({np.nan: None}).to_dict("records")
        front = records[0]
        return {
            "expected_moves": records,
            "front_expected_move": front.get("expected_move"),
            "front_expected_move_pct": front.get("expected_move_pct"),
            "front_expected_move_method": front.get("method"),
        }

    @staticmethod
    def _svi_metadata(chain: pd.DataFrame, spot: float, iv_column: str = "computedIV") -> Dict[str, Any]:
        svi = calibrate_svi_by_expiry(chain, spot, iv_column=iv_column)
        standard_svi = calibrate_svi_by_expiry(
            chain,
            spot,
            iv_column=iv_column,
            weight_column=None,
            use_weight_fallbacks=False,
            loss="linear",
        )
        ssvi = calibrate_ssvi_surface(chain, spot, iv_column=iv_column)
        diagnostics = fit_diagnostics_from_svi(svi)
        standard_diagnostics = fit_diagnostics_from_svi(standard_svi)
        global_diagnostics = fit_diagnostics_from_ssvi(ssvi)
        fit_mode_comparison = DashboardConnector._surface_fit_mode_comparison(
            standard_diagnostics,
            diagnostics,
            global_diagnostics,
        )
        if svi.empty:
            return {
                "svi_smiles": [],
                "fit_diagnostics": diagnostics,
                "standard_svi_smiles": standard_svi.replace({np.nan: None}).to_dict("records"),
                "standard_fit_diagnostics": standard_diagnostics,
                "ssvi_surface": ssvi,
                "global_fit_diagnostics": global_diagnostics,
                "fit_mode_comparison": fit_mode_comparison,
            }
        records = svi.replace({np.nan: None}).to_dict("records")
        return {
            "svi_smiles": records,
            "fit_diagnostics": diagnostics,
            "standard_svi_smiles": standard_svi.replace({np.nan: None}).to_dict("records"),
            "standard_fit_diagnostics": standard_diagnostics,
            "ssvi_surface": ssvi,
            "global_fit_diagnostics": global_diagnostics,
            "fit_mode_comparison": fit_mode_comparison,
            "front_svi_rmse": records[0].get("rmse"),
            "front_svi_mae": records[0].get("mae"),
        }

    @staticmethod
    def _surface_fit_mode_comparison(
        standard_svi: Dict[str, Any],
        robust_svi: Dict[str, Any],
        robust_ssvi: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        modes = [
            ("Standard SVI", standard_svi, "unweighted_linear_loss"),
            ("Robust SVI", robust_svi, "weighted_quote_reliability_soft_l1"),
            ("Robust SSVI", robust_ssvi, "weighted_global_ssvi_soft_l1"),
        ]
        rows: list[Dict[str, Any]] = []
        for name, diagnostics, policy in modes:
            residuals = diagnostics.get("residual_diagnostics") or {}
            rows.append(
                {
                    "mode": name,
                    "model": diagnostics.get("model"),
                    "status": diagnostics.get("status", "fitted" if diagnostics.get("points") else "unavailable"),
                    "fit_policy": policy,
                    "fitted_expiries": diagnostics.get("fitted_expiries"),
                    "points": diagnostics.get("points"),
                    "rmse": diagnostics.get("rmse"),
                    "weighted_rmse": diagnostics.get("weighted_rmse"),
                    "mae": diagnostics.get("mae"),
                    "max_error": diagnostics.get("max_error"),
                    "weight_mode": diagnostics.get("weight_mode"),
                    "loss_mode": diagnostics.get("loss_mode"),
                    "clipped_count": residuals.get("clipped_count"),
                    "downweighted_count": residuals.get("downweighted_count"),
                    "clip_threshold_abs_residual": residuals.get("clip_threshold_abs_residual"),
                    "rmse_after_clipping": residuals.get("rmse_after_clipping"),
                    "constraints_passed": diagnostics.get("constraints_passed"),
                }
            )
        return rows

    @staticmethod
    def _heston_metadata(chain: pd.DataFrame, spot: float, iv_column: str = "computedIV") -> Dict[str, Any]:
        heston = calibrate_heston_research(chain, spot, iv_column=iv_column)
        return {
            "heston_research": heston,
            "heston_research_status": heston.get("status"),
            "heston_research_rmse": heston.get("rmse"),
            "heston_research_points": heston.get("points"),
            "heston_research_warning": " | ".join(str(item) for item in heston.get("warnings", [])[:2]),
        }

    @staticmethod
    def _sabr_metadata(
        symbol: str,
        chain: pd.DataFrame,
        spot: float,
        iv_column: str = "computedIV",
    ) -> Dict[str, Any]:
        sabr = calibrate_sabr_by_expiry(chain, spot, symbol=symbol, iv_column=iv_column)
        return {
            "sabr": sabr,
            "sabr_status": sabr.get("status"),
            "sabr_rmse": sabr.get("rmse"),
            "sabr_points": sabr.get("points"),
        }

    @staticmethod
    def _local_vol_metadata(
        strikes: np.ndarray,
        expiries: np.ndarray,
        vols: np.ndarray,
        spot: float,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        quality_score = metadata.get("surface_quality_score") or metadata.get("data_quality_score")
        local_vol = dupire_local_vol_surface(
            strikes,
            expiries,
            vols,
            spot,
            quality_score=quality_score,
            smoothing_meta=metadata.get("surface_smoothing"),
        )
        return {"local_volatility": local_vol}

    def _iv_history_metadata(
        self,
        symbol: str,
        chain: pd.DataFrame,
        spot: float,
        iv_column: str = "computedIV",
    ) -> Dict[str, Any]:
        current_iv = atm_iv_from_chain(chain, spot)
        if current_iv is None and iv_column != "computedIV":
            current_iv = atm_iv_from_chain(chain.rename(columns={iv_column: "computedIV"}), spot)
        history = iv_rank_percentile_from_snapshots(symbol, current_iv, self.snapshot_dir)
        return {
            "iv_history": history,
            "iv_rank": history.get("iv_rank"),
            "iv_percentile": history.get("iv_percentile"),
            "iv_history_observations": history.get("observations"),
        }

    def _surface_change_metadata(
        self,
        symbol: str,
        chain: pd.DataFrame,
        spot: float,
        metadata: Dict[str, Any],
        iv_column: str = "computedIV",
    ) -> Dict[str, Any]:
        change = surface_change_analytics(
            symbol,
            chain,
            spot,
            self.snapshot_dir,
            iv_column=iv_column,
            current_timestamp=metadata.get("timestamp") or metadata.get("spot_timestamp") or datetime.now(),
            vol_of_vol_history=(metadata.get("iv_history") or {}).get("history"),
        )
        atm_change = change.get("atm_change") or {}
        vol_of_vol = change.get("vol_of_vol") or {}
        return {
            "surface_change": change,
            "surface_change_available": change.get("available"),
            "surface_change_points": change.get("matched_points"),
            "surface_tape": change.get("tape") or {},
            "surface_tape_available": (change.get("tape") or {}).get("available"),
            "surface_tape_snapshots": (change.get("tape") or {}).get("snapshot_count"),
            "surface_change_heatmaps": change.get("heatmaps") or {},
            "surface_change_heatmap_available": (change.get("heatmaps") or {}).get("available"),
            "atm_iv_change": atm_change.get("iv_change"),
            "atm_iv_change_pct": atm_change.get("iv_change_pct"),
            "snapshot_vol_of_vol": vol_of_vol.get("snapshot_vol_of_vol"),
            "annualized_vol_of_vol": vol_of_vol.get("annualized_vol_of_vol"),
            "vol_of_vol_observations": vol_of_vol.get("observations"),
        }

    @staticmethod
    def _rich_cheap_metadata(
        chain: pd.DataFrame,
        metadata: Dict[str, Any],
        iv_column: str = "computedIV",
    ) -> Dict[str, Any]:
        scanner = rich_cheap_scanner(chain, metadata.get("svi_smiles") or [], iv_column=iv_column)
        return {
            "rich_cheap_scanner": scanner,
            "rich_cheap_scanner_available": scanner.get("available"),
            "rich_cheap_candidates": scanner.get("candidate_count"),
            "rich_cheap_rich_count": scanner.get("rich_count"),
            "rich_cheap_cheap_count": scanner.get("cheap_count"),
        }

    @staticmethod
    def _surface_shock_metadata(
        chain: pd.DataFrame,
        spot: float,
        iv_column: str = "computedIV",
    ) -> Dict[str, Any]:
        shock_chain = chain
        if iv_column != "computedIV" and iv_column in chain:
            shock_chain = chain.rename(columns={iv_column: "computedIV"})
        shocks = surface_shock_scenarios(shock_chain, spot)
        return {
            "surface_shocks": shocks,
            "surface_shock_available": shocks.get("available"),
            "surface_shock_contracts": shocks.get("base_contracts"),
            "surface_shock_position_assumption": shocks.get("position_assumption"),
        }

    @staticmethod
    def _event_metadata(
        event_snapshot: Any,
        chain: pd.DataFrame,
        dividend_assumption: Any,
        corporate_actions: Any,
    ) -> Dict[str, Any]:
        extra_events = _supplemental_market_events(
            event_snapshot.symbol,
            event_snapshot.as_of,
            dividend_assumption,
            corporate_actions,
        )
        metadata = event_snapshot.metadata_dict()
        combined: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for event in metadata.get("events") or []:
            combined[(event["symbol"], event["event_type"], event["event_date"])] = dict(event)
        for event in extra_events:
            payload = event.as_dict()
            combined[(payload["symbol"], payload["event_type"], payload["event_date"])] = payload

        metadata["events"] = sorted(
            combined.values(),
            key=lambda item: (str(item.get("event_date")), str(item.get("event_type")), str(item.get("symbol"))),
        )
        metadata["event_count"] = len(metadata["events"])
        metadata["expiry_events"] = (
            expiry_event_metadata(chain["expiration"], event_snapshot, extra_events)
            if not chain.empty and "expiration" in chain
            else {}
        )
        metadata["event_expiry_count"] = len(metadata["expiry_events"])
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
        no_arbitrage_excluded = 0
        fit_excluded = 0
        if "fitEligible" in out:
            fit_mask = out["fitEligible"].fillna(False).astype(bool)
            fit_excluded = int((~fit_mask).sum())
            if "noArbitrageViolation" in out:
                violation_mask = out["noArbitrageViolation"].fillna(False).astype(bool)
                no_arbitrage_excluded = int((~fit_mask & violation_mask).sum())
            out = out[fit_mask].copy()
        elif "noArbitrageViolation" in out:
            violation_mask = out["noArbitrageViolation"].fillna(False).astype(bool)
            no_arbitrage_excluded = int(violation_mask.sum())
            fit_excluded = no_arbitrage_excluded
            out = out[~violation_mask].copy()
        out.attrs["fit_eligible_count"] = int(len(out))
        out.attrs["fit_excluded_count"] = fit_excluded
        out.attrs["no_arbitrage_excluded_count"] = no_arbitrage_excluded
        out["impliedVolatility"] = pd.to_numeric(out["computedIV"], errors="coerce")
        return out

    @staticmethod
    def _apply_parity_checks(chain: pd.DataFrame, spot: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Flag obvious put-call parity violations by expiration and strike."""
        if chain.empty:
            return chain.copy(), {
                "parity_pairs_checked": 0,
                "parity_violation_count": 0,
                "parity_violation_rows": 0,
                "parity_violations": [],
            }

        df = chain.copy()
        df["parityViolation"] = False
        df["parityError"] = np.nan
        df["parityTheoreticalDiff"] = np.nan
        df["parityObservedDiff"] = np.nan

        required = {"type", "strike", "expiration", "selectedMarketPrice"}
        if not required.issubset(df.columns):
            return df, {
                "parity_pairs_checked": 0,
                "parity_violation_count": 0,
                "parity_violation_rows": 0,
                "parity_violations": [],
            }

        work = df.copy()
        work["expiration_norm"] = pd.to_datetime(work["expiration"], errors="coerce").dt.normalize()
        work["strike_num"] = pd.to_numeric(work["strike"], errors="coerce")
        work["selected_price_num"] = pd.to_numeric(work["selectedMarketPrice"], errors="coerce")

        pairs_checked = 0
        violations: List[Dict[str, Any]] = []
        for (expiration, strike), group in work.groupby(["expiration_norm", "strike_num"], dropna=True):
            calls = group[group["type"].astype(str).str.lower() == "call"]
            puts = group[group["type"].astype(str).str.lower() == "put"]
            if calls.empty or puts.empty:
                continue
            call_price = float(calls["selected_price_num"].dropna().median()) if calls["selected_price_num"].notna().any() else np.nan
            put_price = float(puts["selected_price_num"].dropna().median()) if puts["selected_price_num"].notna().any() else np.nan
            if not np.isfinite(call_price) or not np.isfinite(put_price):
                continue

            representative = group.iloc[0]
            time_to_expiry = _float_or_nan(representative.get("time_to_expiry"))
            if not np.isfinite(time_to_expiry):
                days_to_expiry = _float_or_nan(representative.get("daysToExpiration"))
                time_to_expiry = days_to_expiry / 365.0 if np.isfinite(days_to_expiry) else np.nan
            rate = _float_or_nan(representative.get("riskFreeRate"))
            dividend = _float_or_nan(representative.get("effectiveDividendYield"))
            if not np.isfinite(dividend):
                dividend = _float_or_nan(representative.get("dividendYield"))
            if not all(np.isfinite(value) for value in (time_to_expiry, rate, dividend)):
                continue

            pairs_checked += 1
            observed = call_price - put_price
            theoretical = spot * np.exp(-dividend * time_to_expiry) - float(strike) * np.exp(-rate * time_to_expiry)
            error = observed - theoretical
            tolerance = max(1.0, 0.20 * max(abs(call_price), abs(put_price), abs(theoretical), 1.0))
            pair_index = group.index
            df.loc[pair_index, "parityError"] = error
            df.loc[pair_index, "parityTheoreticalDiff"] = theoretical
            df.loc[pair_index, "parityObservedDiff"] = observed
            if abs(error) > tolerance:
                df.loc[pair_index, "parityViolation"] = True
                violations.append(
                    {
                        "expiration": expiration.date().isoformat() if pd.notna(expiration) else None,
                        "strike": float(strike),
                        "call_price": call_price,
                        "put_price": put_price,
                        "observed_call_put_diff": observed,
                        "theoretical_call_put_diff": theoretical,
                        "parity_error": error,
                        "tolerance": tolerance,
                    }
                )

        return df, {
            "parity_pairs_checked": pairs_checked,
            "parity_violation_count": len(violations),
            "parity_violation_rows": int(df["parityViolation"].sum()),
            "parity_violations": violations[:20],
        }

    @staticmethod
    def _data_quality_metadata(chain: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Build expiry-level and chain-level quality diagnostics."""
        expiry_quality = _copy_expiry_quality(metadata.get("expiry_quality"))
        if not chain.empty and "expiration" in chain.columns:
            expirations = pd.to_datetime(chain["expiration"], errors="coerce").dt.normalize()
            for expiry, group in chain.groupby(expirations, dropna=True):
                key = expiry.date().isoformat()
                entry = expiry_quality.setdefault(
                    key,
                    {"valid_quotes": 0, "rejected_quotes": 0, "reason_buckets": {}},
                )
                entry["valid_quotes"] = int(len(group))
                reason_buckets = dict(entry.get("reason_buckets") or {})

                if "computedIV" in group:
                    computed = pd.to_numeric(group["computedIV"], errors="coerce")
                    failed = int(computed.isna().sum())
                    if failed:
                        reason_buckets["computed_iv_failed"] = failed
                    entry["computed_iv_count"] = int(computed.notna().sum())

                if "parityViolation" in group:
                    parity_rows = int(group["parityViolation"].fillna(False).astype(bool).sum())
                    if parity_rows:
                        reason_buckets["parity_violation"] = parity_rows
                if "noArbitrageViolation" in group:
                    arbitrage_rows = int(group["noArbitrageViolation"].fillna(False).astype(bool).sum())
                    if arbitrage_rows:
                        reason_buckets["no_arbitrage_violation"] = arbitrage_rows

                entry["reason_buckets"] = {
                    reason: int(count) for reason, count in reason_buckets.items() if count
                }
                entry["rejected_quotes"] = int(entry.get("rejected_quotes") or 0)
                entry["score"] = _quality_score(
                    entry["valid_quotes"],
                    entry["rejected_quotes"],
                    entry["reason_buckets"],
                )

        valid_rows = int(metadata.get("valid_rows") or (len(chain) if not chain.empty else 0))
        rejected_rows = int(metadata.get("rejected_rows") or 0)
        reason_buckets = dict(metadata.get("rejection_reasons") or {})
        computed_failed = int(metadata.get("computed_iv_failed_count") or 0)
        if computed_failed:
            reason_buckets["computed_iv_failed"] = computed_failed
        parity_rows = int(metadata.get("parity_violation_rows") or 0)
        if parity_rows:
            reason_buckets["parity_violation"] = parity_rows
        arbitrage_rows = int(metadata.get("no_arbitrage_violation_rows") or 0)
        if arbitrage_rows:
            reason_buckets["no_arbitrage_violation"] = arbitrage_rows
        score = _quality_score(valid_rows, rejected_rows, reason_buckets)
        return {
            "data_quality_score": score,
            "quality_score": score,
            "quality_reason_buckets": {
                reason: int(count) for reason, count in reason_buckets.items() if count
            },
            "expiry_quality": dict(sorted(expiry_quality.items())),
        }

    @staticmethod
    def _surface_quality_metadata(surface_chain: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize the quotes actually used to fit the displayed surface."""
        expiry_quality = _copy_expiry_quality(metadata.get("expiry_quality"))
        surface_expiry_counts: Dict[str, int] = {}
        if not surface_chain.empty and "expiration" in surface_chain.columns:
            expirations = pd.to_datetime(surface_chain["expiration"], errors="coerce").dt.normalize()
            for expiry, group in surface_chain.groupby(expirations, dropna=True):
                key = expiry.date().isoformat()
                surface_expiry_counts[key] = int(len(group))
                entry = expiry_quality.setdefault(
                    key,
                    {"valid_quotes": 0, "rejected_quotes": 0, "reason_buckets": {}},
                )
                entry["surface_quotes"] = int(len(group))
                entry["score"] = _quality_score(
                    int(entry.get("valid_quotes") or 0),
                    int(entry.get("rejected_quotes") or 0),
                    dict(entry.get("reason_buckets") or {}),
                )

        valid_rows = int(metadata.get("valid_rows") or len(surface_chain))
        rejected_rows = int(metadata.get("rejected_rows") or 0)
        reason_buckets = dict(metadata.get("quality_reason_buckets") or metadata.get("rejection_reasons") or {})
        score = _quality_score(valid_rows, rejected_rows, reason_buckets)
        surface_quality = {
            "score": score,
            "valid_quotes": valid_rows,
            "rejected_quotes": rejected_rows,
            "surface_quotes": int(len(surface_chain)),
            "fit_eligible_count": int(surface_chain.attrs.get("fit_eligible_count", len(surface_chain))),
            "fit_excluded_count": int(
                surface_chain.attrs.get(
                    "fit_excluded_count",
                    metadata.get("fit_excluded_count") or 0,
                )
            ),
            "no_arbitrage_excluded_quotes": int(surface_chain.attrs.get("no_arbitrage_excluded_count", 0)),
            "reason_buckets": {reason: int(count) for reason, count in reason_buckets.items() if count},
            "expiries": surface_expiry_counts,
        }
        return {
            "surface_quality_score": score,
            "surface_quality": surface_quality,
            "fit_eligible_count": int(surface_quality["fit_eligible_count"]),
            "fit_excluded_count": int(surface_quality["fit_excluded_count"]),
            "no_arbitrage_excluded_count": int(surface_chain.attrs.get("no_arbitrage_excluded_count", 0)),
            "expiry_quality": dict(sorted(expiry_quality.items())),
        }

    def _fit_filter_metadata(self) -> Dict[str, Any]:
        filters = asdict(self.fit_filters)
        return {
            "fit_filters": filters,
            "fit_filter_preset": filters["preset"],
            "fit_max_bid_ask_spread_pct": filters["max_bid_ask_spread_pct"],
            "fit_max_quote_age_days": filters["max_quote_age_days"],
            "fit_min_volume": filters["min_volume"],
            "fit_min_open_interest": filters["min_open_interest"],
            "fit_moneyness_min": filters["moneyness_min"],
            "fit_moneyness_max": filters["moneyness_max"],
            "fit_max_raw_iv": filters["max_raw_iv"],
            "fit_no_arbitrage_policy": filters["no_arbitrage_policy"],
            "fit_last_only_policy": filters["last_only_policy"],
        }

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
            "pricing_model": self.pricing_model,
            "pricing_model_label": pricing_model_metadata(self.pricing_model)["pricing_model_label"],
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


def _copy_expiry_quality(value: Any) -> Dict[str, Dict[str, Any]]:
    if not value:
        return {}
    items = value.items() if isinstance(value, dict) else value
    out: Dict[str, Dict[str, Any]] = {}
    for expiry, payload in items:
        entry = dict(payload or {})
        entry["valid_quotes"] = int(entry.get("valid_quotes") or 0)
        entry["rejected_quotes"] = int(entry.get("rejected_quotes") or 0)
        entry["reason_buckets"] = {
            str(reason): int(count)
            for reason, count in dict(entry.get("reason_buckets") or {}).items()
            if count
        }
        out[str(expiry)] = entry
    return out


def _supplemental_market_events(
    symbol: str,
    as_of: datetime,
    dividend_assumption: Any,
    corporate_actions: Any,
) -> tuple[MarketEvent, ...]:
    events: Dict[tuple[str, str, str], MarketEvent] = {}
    start = as_of.date()
    horizon = start + timedelta(days=370)

    for event in getattr(dividend_assumption, "events", ()) or ():
        event_date = getattr(event, "ex_date", None)
        if event_date is None or not (start <= event_date <= horizon):
            continue
        amount = getattr(event, "amount", None)
        currency = getattr(event, "currency", "USD")
        market_event = MarketEvent(
            symbol=symbol.upper(),
            event_type="dividend",
            event_date=event_date,
            description=f"Cash dividend {amount:g} {currency}" if amount is not None else "Cash dividend",
            source=getattr(dividend_assumption, "source", "dividend assumptions"),
        )
        events[(market_event.symbol, market_event.event_type, market_event.event_date.isoformat())] = market_event

    for action in getattr(corporate_actions, "upcoming", lambda: ())():
        event_date = getattr(action, "effective_date", None)
        if event_date is None or not (start <= event_date <= horizon):
            continue
        action_type = str(getattr(action, "action_type", "other") or "other")
        event_type = "dividend" if action_type == "dividend" else "corporate_action"
        market_event = MarketEvent(
            symbol=symbol.upper(),
            event_type=event_type,
            event_date=event_date,
            description=str(getattr(action, "description", action_type.title())),
            source=str(getattr(action, "source", "corporate-action assumptions")),
        )
        events[(market_event.symbol, market_event.event_type, market_event.event_date.isoformat())] = market_event

    return tuple(sorted(events.values(), key=lambda event: (event.event_date, event.event_type, event.symbol)))


def _quality_score(valid_quotes: int, rejected_quotes: int, reason_buckets: Dict[str, int]) -> float:
    valid = max(0, int(valid_quotes or 0))
    rejected = max(0, int(rejected_quotes or 0))
    total = valid + rejected
    if total <= 0:
        return 0.0

    score = 100.0 * valid / total
    score -= 20.0 * int(reason_buckets.get("computed_iv_failed", 0)) / max(valid, 1)
    score -= 15.0 * int(reason_buckets.get("parity_violation", 0)) / max(valid, 1)
    score -= 20.0 * int(reason_buckets.get("no_arbitrage_violation", 0)) / max(valid, 1)
    return round(float(min(100.0, max(0.0, score))), 1)
