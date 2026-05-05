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
    mark: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    raw_iv: Optional[float] = None
    computed_iv: Optional[float] = None
    selected_market_price: Optional[float] = None
    selected_price_source: Optional[str] = None
    iv_input: Optional[str] = None
    parity_violation: Optional[bool] = None
    parity_error: Optional[float] = None
    parity_theoretical_diff: Optional[float] = None
    parity_observed_diff: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    risk_free_rate: Optional[float] = None
    discount_factor: Optional[float] = None
    forward_price: Optional[float] = None
    forward_moneyness: Optional[float] = None
    log_moneyness: Optional[float] = None
    dividend_yield: Optional[float] = None
    effective_dividend_yield: Optional[float] = None
    discrete_dividend_amount: Optional[float] = None
    discrete_dividend_pv: Optional[float] = None
    discrete_dividend_count: Optional[int] = None
    quote_timestamp: Optional[datetime] = None
    quote_age_seconds: Optional[float] = None
    is_stale_quote: Optional[bool] = None
    is_crossed_market: Optional[bool] = None
    is_locked_market: Optional[bool] = None
    quote_quality: Optional[str] = None
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
            mark=_float_or_none(row.get("mark")),
            last=_float_or_none(row.get("last")),
            volume=_int_or_none(row.get("volume")),
            open_interest=_int_or_none(row.get("openInterest")),
            raw_iv=_float_or_none(row.get("impliedVolatility")),
            computed_iv=_float_or_none(row.get("computedIV")),
            selected_market_price=_float_or_none(row.get("selectedMarketPrice")),
            selected_price_source=_str_or_none(row.get("selectedPriceSource")),
            iv_input=_str_or_none(row.get("ivInput")),
            parity_violation=_bool_or_none(row.get("parityViolation")),
            parity_error=_float_or_none(row.get("parityError")),
            parity_theoretical_diff=_float_or_none(row.get("parityTheoreticalDiff")),
            parity_observed_diff=_float_or_none(row.get("parityObservedDiff")),
            delta=_float_or_none(row.get("delta")),
            gamma=_float_or_none(row.get("gamma")),
            theta=_float_or_none(row.get("theta")),
            vega=_float_or_none(row.get("vega")),
            rho=_float_or_none(row.get("rho")),
            risk_free_rate=_float_or_none(row.get("riskFreeRate")),
            discount_factor=_float_or_none(row.get("discountFactor")),
            forward_price=_float_or_none(row.get("forwardPrice")),
            forward_moneyness=_float_or_none(row.get("forwardMoneyness")),
            log_moneyness=_float_or_none(row.get("logMoneyness")),
            dividend_yield=_float_or_none(row.get("dividendYield")),
            effective_dividend_yield=_float_or_none(row.get("effectiveDividendYield")),
            discrete_dividend_amount=_float_or_none(row.get("discreteDividendAmount")),
            discrete_dividend_pv=_float_or_none(row.get("discreteDividendPV")),
            discrete_dividend_count=_int_or_none(row.get("discreteDividendCount")),
            quote_timestamp=quote_ts.to_pydatetime() if pd.notna(quote_ts) else None,
            quote_age_seconds=_float_or_none(row.get("quoteAgeSeconds")),
            is_stale_quote=_bool_or_none(row.get("isStaleQuote")),
            is_crossed_market=_bool_or_none(row.get("isCrossedMarket")),
            is_locked_market=_bool_or_none(row.get("isLockedMarket")),
            quote_quality=_str_or_none(row.get("quoteQuality")),
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
            "mark": self.mark,
            "last": self.last,
            "volume": self.volume,
            "openInterest": self.open_interest,
            "impliedVolatility": self.raw_iv,
            "computedIV": self.computed_iv,
            "selectedMarketPrice": self.selected_market_price,
            "selectedPriceSource": self.selected_price_source,
            "ivInput": self.iv_input,
            "parityViolation": self.parity_violation,
            "parityError": self.parity_error,
            "parityTheoreticalDiff": self.parity_theoretical_diff,
            "parityObservedDiff": self.parity_observed_diff,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "rho": self.rho,
            "riskFreeRate": self.risk_free_rate,
            "discountFactor": self.discount_factor,
            "forwardPrice": self.forward_price,
            "forwardMoneyness": self.forward_moneyness,
            "logMoneyness": self.log_moneyness,
            "dividendYield": self.dividend_yield,
            "effectiveDividendYield": self.effective_dividend_yield,
            "discreteDividendAmount": self.discrete_dividend_amount,
            "discreteDividendPV": self.discrete_dividend_pv,
            "discreteDividendCount": self.discrete_dividend_count,
            "quoteTimestamp": self.quote_timestamp,
            "quoteAgeSeconds": self.quote_age_seconds,
            "isStaleQuote": self.is_stale_quote,
            "isCrossedMarket": self.is_crossed_market,
            "isLockedMarket": self.is_locked_market,
            "quoteQuality": self.quote_quality,
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
    risk_free_rate_source: Optional[str] = None
    risk_free_rate_mode: Optional[str] = None
    risk_free_rate_timestamp: Optional[datetime] = None
    risk_free_rate_fallback_reason: Optional[str] = None
    risk_free_rate_curve: tuple[tuple[int, float], ...] = field(default_factory=tuple)
    expiry_rates: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    risk_free_rate_30d: Optional[float] = None
    risk_free_rate_min: Optional[float] = None
    risk_free_rate_max: Optional[float] = None
    risk_free_rate_median: Optional[float] = None
    expiry_forwards: tuple[tuple[str, dict[str, float]], ...] = field(default_factory=tuple)
    forward_price_min: Optional[float] = None
    forward_price_max: Optional[float] = None
    forward_price_median: Optional[float] = None
    discount_factor_min: Optional[float] = None
    discount_factor_max: Optional[float] = None
    discount_factor_median: Optional[float] = None
    dividend_source: Optional[str] = None
    dividend_mode: Optional[str] = None
    dividend_timestamp: Optional[datetime] = None
    dividend_fallback_reason: Optional[str] = None
    annual_dividend_yield: Optional[float] = None
    dividend_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    expiry_dividends: tuple[tuple[str, dict[str, float | int]], ...] = field(default_factory=tuple)
    effective_dividend_yield_30d: Optional[float] = None
    effective_dividend_yield_min: Optional[float] = None
    effective_dividend_yield_max: Optional[float] = None
    effective_dividend_yield_median: Optional[float] = None
    corporate_action_source: Optional[str] = None
    corporate_action_mode: Optional[str] = None
    corporate_action_timestamp: Optional[datetime] = None
    corporate_action_fallback_reason: Optional[str] = None
    corporate_actions: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    upcoming_corporate_actions: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    expiry_corporate_actions: tuple[tuple[str, list[dict[str, Any]]], ...] = field(default_factory=tuple)
    corporate_action_warning_count: int = 0
    corporate_action_warnings: tuple[str, ...] = field(default_factory=tuple)
    stale_quote_count: int = 0
    last_only_quote_count: int = 0
    zero_bid_ask_count: int = 0
    crossed_market_count: int = 0
    locked_market_count: int = 0
    crossed_locked_rejected_count: int = 0
    stale_last_only_rejected_count: int = 0
    min_open_interest: int = 0
    min_volume: int = 0
    max_bid_ask_spread_pct: Optional[float] = None
    liquidity_filtered_count: int = 0
    low_open_interest_rejected_count: int = 0
    low_volume_rejected_count: int = 0
    wide_spread_rejected_count: int = 0
    old_quote_rejected_count: int = 0
    rejection_reasons: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    data_quality_score: Optional[float] = None
    quality_score: Optional[float] = None
    quality_reason_buckets: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    expiry_quality: tuple[tuple[str, dict[str, Any]], ...] = field(default_factory=tuple)
    max_quote_age_days: Optional[int] = None
    option_price_source: str = "mark"
    computed_iv_count: int = 0
    computed_iv_failed_count: int = 0
    parity_pairs_checked: int = 0
    parity_violation_count: int = 0
    parity_violation_rows: int = 0
    parity_violations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
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
            risk_free_rate_source=metadata.get("risk_free_rate_source"),
            risk_free_rate_mode=metadata.get("risk_free_rate_mode"),
            risk_free_rate_timestamp=_datetime_or_none(metadata.get("risk_free_rate_timestamp")),
            risk_free_rate_fallback_reason=metadata.get("risk_free_rate_fallback_reason"),
            risk_free_rate_curve=_rate_curve_tuple(metadata.get("risk_free_rate_curve")),
            expiry_rates=_expiry_rates_tuple(metadata.get("expiry_rates")),
            risk_free_rate_30d=_float_or_none(metadata.get("risk_free_rate_30d")),
            risk_free_rate_min=_float_or_none(metadata.get("risk_free_rate_min")),
            risk_free_rate_max=_float_or_none(metadata.get("risk_free_rate_max")),
            risk_free_rate_median=_float_or_none(metadata.get("risk_free_rate_median")),
            expiry_forwards=_nested_float_metadata_tuple(metadata.get("expiry_forwards")),
            forward_price_min=_float_or_none(metadata.get("forward_price_min")),
            forward_price_max=_float_or_none(metadata.get("forward_price_max")),
            forward_price_median=_float_or_none(metadata.get("forward_price_median")),
            discount_factor_min=_float_or_none(metadata.get("discount_factor_min")),
            discount_factor_max=_float_or_none(metadata.get("discount_factor_max")),
            discount_factor_median=_float_or_none(metadata.get("discount_factor_median")),
            dividend_source=metadata.get("dividend_source"),
            dividend_mode=metadata.get("dividend_mode"),
            dividend_timestamp=_datetime_or_none(metadata.get("dividend_timestamp")),
            dividend_fallback_reason=metadata.get("dividend_fallback_reason"),
            annual_dividend_yield=_float_or_none(metadata.get("annual_dividend_yield")),
            dividend_events=_dict_tuple(metadata.get("dividend_events")),
            expiry_dividends=_nested_metadata_tuple(metadata.get("expiry_dividends")),
            effective_dividend_yield_30d=_float_or_none(metadata.get("effective_dividend_yield_30d")),
            effective_dividend_yield_min=_float_or_none(metadata.get("effective_dividend_yield_min")),
            effective_dividend_yield_max=_float_or_none(metadata.get("effective_dividend_yield_max")),
            effective_dividend_yield_median=_float_or_none(metadata.get("effective_dividend_yield_median")),
            corporate_action_source=metadata.get("corporate_action_source"),
            corporate_action_mode=metadata.get("corporate_action_mode"),
            corporate_action_timestamp=_datetime_or_none(metadata.get("corporate_action_timestamp")),
            corporate_action_fallback_reason=metadata.get("corporate_action_fallback_reason"),
            corporate_actions=_dict_tuple(metadata.get("corporate_actions")),
            upcoming_corporate_actions=_dict_tuple(metadata.get("upcoming_corporate_actions")),
            expiry_corporate_actions=_list_metadata_tuple(metadata.get("expiry_corporate_actions")),
            corporate_action_warning_count=int(metadata.get("corporate_action_warning_count") or 0),
            corporate_action_warnings=tuple(str(item) for item in metadata.get("corporate_action_warnings") or ()),
            stale_quote_count=int(metadata.get("stale_quote_count") or 0),
            last_only_quote_count=int(metadata.get("last_only_quote_count") or 0),
            zero_bid_ask_count=int(metadata.get("zero_bid_ask_count") or 0),
            crossed_market_count=int(metadata.get("crossed_market_count") or 0),
            locked_market_count=int(metadata.get("locked_market_count") or 0),
            crossed_locked_rejected_count=int(metadata.get("crossed_locked_rejected_count") or 0),
            stale_last_only_rejected_count=int(metadata.get("stale_last_only_rejected_count") or 0),
            min_open_interest=int(metadata.get("min_open_interest") or 0),
            min_volume=int(metadata.get("min_volume") or 0),
            max_bid_ask_spread_pct=_float_or_none(metadata.get("max_bid_ask_spread_pct")),
            liquidity_filtered_count=int(metadata.get("liquidity_filtered_count") or 0),
            low_open_interest_rejected_count=int(metadata.get("low_open_interest_rejected_count") or 0),
            low_volume_rejected_count=int(metadata.get("low_volume_rejected_count") or 0),
            wide_spread_rejected_count=int(metadata.get("wide_spread_rejected_count") or 0),
            old_quote_rejected_count=int(metadata.get("old_quote_rejected_count") or 0),
            rejection_reasons=_int_metadata_tuple(metadata.get("rejection_reasons")),
            data_quality_score=_float_or_none(metadata.get("data_quality_score")),
            quality_score=_float_or_none(metadata.get("quality_score")),
            quality_reason_buckets=_int_metadata_tuple(metadata.get("quality_reason_buckets")),
            expiry_quality=_nested_any_metadata_tuple(metadata.get("expiry_quality")),
            max_quote_age_days=_int_or_none(metadata.get("max_quote_age_days")),
            option_price_source=str(metadata.get("option_price_source") or "mark"),
            computed_iv_count=int(metadata.get("computed_iv_count") or 0),
            computed_iv_failed_count=int(metadata.get("computed_iv_failed_count") or 0),
            parity_pairs_checked=int(metadata.get("parity_pairs_checked") or 0),
            parity_violation_count=int(metadata.get("parity_violation_count") or 0),
            parity_violation_rows=int(metadata.get("parity_violation_rows") or 0),
            parity_violations=_dict_tuple(metadata.get("parity_violations")),
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
            "risk_free_rate_source": self.risk_free_rate_source,
            "risk_free_rate_mode": self.risk_free_rate_mode,
            "risk_free_rate_timestamp": self.risk_free_rate_timestamp,
            "risk_free_rate_fallback_reason": self.risk_free_rate_fallback_reason,
            "risk_free_rate_curve": [
                {"tenor_days": tenor_days, "rate": rate} for tenor_days, rate in self.risk_free_rate_curve
            ],
            "expiry_rates": dict(self.expiry_rates),
            "risk_free_rate_30d": self.risk_free_rate_30d,
            "risk_free_rate_min": self.risk_free_rate_min,
            "risk_free_rate_max": self.risk_free_rate_max,
            "risk_free_rate_median": self.risk_free_rate_median,
            "expiry_forwards": dict(self.expiry_forwards),
            "forward_price_min": self.forward_price_min,
            "forward_price_max": self.forward_price_max,
            "forward_price_median": self.forward_price_median,
            "discount_factor_min": self.discount_factor_min,
            "discount_factor_max": self.discount_factor_max,
            "discount_factor_median": self.discount_factor_median,
            "dividend_source": self.dividend_source,
            "dividend_mode": self.dividend_mode,
            "dividend_timestamp": self.dividend_timestamp,
            "dividend_fallback_reason": self.dividend_fallback_reason,
            "annual_dividend_yield": self.annual_dividend_yield,
            "dividend_events": list(self.dividend_events),
            "expiry_dividends": dict(self.expiry_dividends),
            "effective_dividend_yield_30d": self.effective_dividend_yield_30d,
            "effective_dividend_yield_min": self.effective_dividend_yield_min,
            "effective_dividend_yield_max": self.effective_dividend_yield_max,
            "effective_dividend_yield_median": self.effective_dividend_yield_median,
            "corporate_action_source": self.corporate_action_source,
            "corporate_action_mode": self.corporate_action_mode,
            "corporate_action_timestamp": self.corporate_action_timestamp,
            "corporate_action_fallback_reason": self.corporate_action_fallback_reason,
            "corporate_actions": list(self.corporate_actions),
            "upcoming_corporate_actions": list(self.upcoming_corporate_actions),
            "expiry_corporate_actions": dict(self.expiry_corporate_actions),
            "corporate_action_warning_count": self.corporate_action_warning_count,
            "corporate_action_warnings": list(self.corporate_action_warnings),
            "stale_quote_count": self.stale_quote_count,
            "last_only_quote_count": self.last_only_quote_count,
            "zero_bid_ask_count": self.zero_bid_ask_count,
            "crossed_market_count": self.crossed_market_count,
            "locked_market_count": self.locked_market_count,
            "crossed_locked_rejected_count": self.crossed_locked_rejected_count,
            "stale_last_only_rejected_count": self.stale_last_only_rejected_count,
            "min_open_interest": self.min_open_interest,
            "min_volume": self.min_volume,
            "max_bid_ask_spread_pct": self.max_bid_ask_spread_pct,
            "liquidity_filtered_count": self.liquidity_filtered_count,
            "low_open_interest_rejected_count": self.low_open_interest_rejected_count,
            "low_volume_rejected_count": self.low_volume_rejected_count,
            "wide_spread_rejected_count": self.wide_spread_rejected_count,
            "old_quote_rejected_count": self.old_quote_rejected_count,
            "rejection_reasons": dict(self.rejection_reasons),
            "data_quality_score": self.data_quality_score,
            "quality_score": self.quality_score,
            "quality_reason_buckets": dict(self.quality_reason_buckets),
            "expiry_quality": dict(self.expiry_quality),
            "max_quote_age_days": self.max_quote_age_days,
            "option_price_source": self.option_price_source,
            "computed_iv_count": self.computed_iv_count,
            "computed_iv_failed_count": self.computed_iv_failed_count,
            "parity_pairs_checked": self.parity_pairs_checked,
            "parity_violation_count": self.parity_violation_count,
            "parity_violation_rows": self.parity_violation_rows,
            "parity_violations": list(self.parity_violations),
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


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None or pd.isna(value):
        return None
    return bool(value)


def _str_or_none(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _datetime_or_none(value: Any) -> Optional[datetime]:
    converted = pd.to_datetime(value, errors="coerce")
    return converted.to_pydatetime() if pd.notna(converted) else None


def _source_delay(mode: Any) -> Optional[timedelta]:
    text = str(mode or "").lower()
    if "delayed" in text:
        return timedelta(minutes=15)
    return None


def _rate_curve_tuple(value: Any) -> tuple[tuple[int, float], ...]:
    if not value:
        return ()
    points = []
    for item in value:
        if isinstance(item, dict):
            tenor = item.get("tenor_days")
            rate = item.get("rate")
        else:
            tenor, rate = item
        if tenor is not None and rate is not None:
            points.append((int(tenor), float(rate)))
    return tuple(points)


def _expiry_rates_tuple(value: Any) -> tuple[tuple[str, float], ...]:
    if not value:
        return ()
    items = value.items() if isinstance(value, dict) else value
    return tuple((str(expiry), float(rate)) for expiry, rate in items)


def _dict_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not value:
        return ()
    return tuple(dict(item) for item in value)


def _nested_metadata_tuple(value: Any) -> tuple[tuple[str, dict[str, float | int]], ...]:
    if not value:
        return ()
    items = value.items() if isinstance(value, dict) else value
    out = []
    for key, payload in items:
        out.append((str(key), dict(payload)))
    return tuple(out)


def _nested_float_metadata_tuple(value: Any) -> tuple[tuple[str, dict[str, float]], ...]:
    if not value:
        return ()
    items = value.items() if isinstance(value, dict) else value
    out = []
    for key, payload in items:
        out.append((str(key), {str(name): float(number) for name, number in dict(payload).items()}))
    return tuple(out)


def _nested_any_metadata_tuple(value: Any) -> tuple[tuple[str, dict[str, Any]], ...]:
    if not value:
        return ()
    items = value.items() if isinstance(value, dict) else value
    out = []
    for key, payload in items:
        out.append((str(key), dict(payload)))
    return tuple(out)


def _list_metadata_tuple(value: Any) -> tuple[tuple[str, list[dict[str, Any]]], ...]:
    if not value:
        return ()
    items = value.items() if isinstance(value, dict) else value
    out = []
    for key, payload in items:
        out.append((str(key), [dict(item) for item in payload]))
    return tuple(out)


def _int_metadata_tuple(value: Any) -> tuple[tuple[str, int], ...]:
    if not value:
        return ()
    items = value.items() if isinstance(value, dict) else value
    out = []
    for key, count in items:
        if count is not None:
            out.append((str(key), int(count)))
    return tuple(out)
