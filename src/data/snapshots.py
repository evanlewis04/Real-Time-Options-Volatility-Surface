"""Local persistence for canonical market-data snapshots."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.models import MarketDataSnapshot, option_quotes_from_frame


DEFAULT_SNAPSHOT_DIR = Path("data/snapshots")


def save_snapshot(snapshot: MarketDataSnapshot, directory: Path | str = DEFAULT_SNAPSHOT_DIR) -> Path:
    """Persist a market snapshot as metadata JSON plus options Parquet."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    stamp = snapshot.spot_timestamp.strftime("%Y%m%d_%H%M%S")
    base = root / f"{snapshot.symbol}_{stamp}"
    options_path = base.with_suffix(".options.parquet")
    metadata_path = base.with_suffix(".metadata.json")

    snapshot.options_frame().to_parquet(options_path, index=False)
    metadata_path.write_text(json.dumps(_snapshot_metadata(snapshot), indent=2), encoding="utf-8")
    return metadata_path


def load_snapshot(metadata_path: Path | str) -> MarketDataSnapshot:
    """Load a persisted market snapshot from a metadata JSON path."""
    path = Path(metadata_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    options_path = Path(metadata["options_path"])
    if not options_path.is_absolute():
        options_path = path.parent / options_path
    frame = pd.read_parquet(options_path) if options_path.exists() else pd.DataFrame()
    quotes = tuple(option_quotes_from_frame(frame))
    expirations = tuple(sorted({quote.expiry for quote in quotes if quote.expiry != datetime.min}))

    return MarketDataSnapshot(
        symbol=metadata["symbol"],
        spot=float(metadata["spot"]),
        spot_timestamp=datetime.fromisoformat(metadata["spot_timestamp"]),
        chain_timestamp=_datetime_from_iso(metadata.get("chain_timestamp")),
        expirations=expirations,
        options=quotes,
        source=metadata.get("source", "unknown"),
        source_delay=_timedelta_from_seconds(metadata.get("source_delay_seconds")),
        cache_age=_timedelta_from_seconds(metadata.get("cache_age_seconds")),
        fallback_reason=metadata.get("fallback_reason"),
        mode=metadata.get("mode", "Unknown"),
        risk_free_rate_source=metadata.get("risk_free_rate_source"),
        risk_free_rate_mode=metadata.get("risk_free_rate_mode"),
        risk_free_rate_timestamp=_datetime_from_iso(metadata.get("risk_free_rate_timestamp")),
        risk_free_rate_fallback_reason=metadata.get("risk_free_rate_fallback_reason"),
        risk_free_rate_curve=tuple(
            (int(point["tenor_days"]), float(point["rate"]))
            for point in metadata.get("risk_free_rate_curve", [])
        ),
        expiry_rates=tuple(
            (str(expiry), float(rate)) for expiry, rate in (metadata.get("expiry_rates") or {}).items()
        ),
        risk_free_rate_30d=_float_or_none(metadata.get("risk_free_rate_30d")),
        risk_free_rate_min=_float_or_none(metadata.get("risk_free_rate_min")),
        risk_free_rate_max=_float_or_none(metadata.get("risk_free_rate_max")),
        risk_free_rate_median=_float_or_none(metadata.get("risk_free_rate_median")),
        dividend_source=metadata.get("dividend_source"),
        dividend_mode=metadata.get("dividend_mode"),
        dividend_timestamp=_datetime_from_iso(metadata.get("dividend_timestamp")),
        dividend_fallback_reason=metadata.get("dividend_fallback_reason"),
        annual_dividend_yield=_float_or_none(metadata.get("annual_dividend_yield")),
        dividend_events=tuple(dict(item) for item in metadata.get("dividend_events", [])),
        expiry_dividends=tuple(
            (str(expiry), dict(payload)) for expiry, payload in (metadata.get("expiry_dividends") or {}).items()
        ),
        effective_dividend_yield_30d=_float_or_none(metadata.get("effective_dividend_yield_30d")),
        effective_dividend_yield_min=_float_or_none(metadata.get("effective_dividend_yield_min")),
        effective_dividend_yield_max=_float_or_none(metadata.get("effective_dividend_yield_max")),
        effective_dividend_yield_median=_float_or_none(metadata.get("effective_dividend_yield_median")),
        corporate_action_source=metadata.get("corporate_action_source"),
        corporate_action_mode=metadata.get("corporate_action_mode"),
        corporate_action_timestamp=_datetime_from_iso(metadata.get("corporate_action_timestamp")),
        corporate_action_fallback_reason=metadata.get("corporate_action_fallback_reason"),
        corporate_actions=tuple(dict(item) for item in metadata.get("corporate_actions", [])),
        upcoming_corporate_actions=tuple(dict(item) for item in metadata.get("upcoming_corporate_actions", [])),
        expiry_corporate_actions=tuple(
            (str(expiry), [dict(item) for item in payload])
            for expiry, payload in (metadata.get("expiry_corporate_actions") or {}).items()
        ),
        corporate_action_warning_count=int(metadata.get("corporate_action_warning_count") or 0),
        corporate_action_warnings=tuple(metadata.get("corporate_action_warnings") or ()),
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
        rejection_reasons=tuple(
            (str(reason), int(count)) for reason, count in (metadata.get("rejection_reasons") or {}).items()
        ),
        data_quality_score=_float_or_none(metadata.get("data_quality_score")),
        quality_score=_float_or_none(metadata.get("quality_score")),
        quality_reason_buckets=tuple(
            (str(reason), int(count)) for reason, count in (metadata.get("quality_reason_buckets") or {}).items()
        ),
        expiry_quality=tuple(
            (str(expiry), dict(payload)) for expiry, payload in (metadata.get("expiry_quality") or {}).items()
        ),
        quote_reliability_summary=dict(metadata.get("quote_reliability_summary") or {}),
        fit_filters=dict(metadata.get("fit_filters") or {}),
        fit_filter_preset=metadata.get("fit_filter_preset"),
        fit_eligible_count=int(metadata.get("fit_eligible_count") or 0),
        fit_excluded_count=int(metadata.get("fit_excluded_count") or 0),
        display_eligible_count=int(metadata.get("display_eligible_count") or 0),
        display_excluded_count=int(metadata.get("display_excluded_count") or 0),
        display_rejection_reason_buckets=tuple(
            (str(reason), int(count))
            for reason, count in (metadata.get("display_rejection_reason_buckets") or {}).items()
        ),
        fit_penalty_reason_buckets=tuple(
            (str(reason), int(count))
            for reason, count in (metadata.get("fit_penalty_reason_buckets") or {}).items()
        ),
        fit_hard_rejection_reason_buckets=tuple(
            (str(reason), int(count))
            for reason, count in (metadata.get("fit_hard_rejection_reason_buckets") or {}).items()
        ),
        expiry_reliability=tuple(
            (str(expiry), dict(payload)) for expiry, payload in (metadata.get("expiry_reliability") or {}).items()
        ),
        max_quote_age_days=_int_or_none(metadata.get("max_quote_age_days")),
        option_price_source=metadata.get("option_price_source", "mark"),
        pricing_model=metadata.get("pricing_model", "bsm_dividends"),
        pricing_model_label=metadata.get("pricing_model_label"),
        pricing_model_assumptions=metadata.get("pricing_model_assumptions"),
        pricing_model_warning=metadata.get("pricing_model_warning"),
        contract_greeks_count=int(metadata.get("contract_greeks_count") or 0),
        second_order_greeks_count=int(metadata.get("second_order_greeks_count") or 0),
        greek_units=tuple(
            (str(name), str(unit)) for name, unit in (metadata.get("greek_units") or {}).items()
        ),
        median_selected_model_residual=_float_or_none(metadata.get("median_selected_model_residual")),
        max_abs_selected_model_residual=_float_or_none(metadata.get("max_abs_selected_model_residual")),
        computed_iv_count=int(metadata.get("computed_iv_count") or 0),
        computed_iv_failed_count=int(metadata.get("computed_iv_failed_count") or 0),
        parity_pairs_checked=int(metadata.get("parity_pairs_checked") or 0),
        parity_violation_count=int(metadata.get("parity_violation_count") or 0),
        parity_violation_rows=int(metadata.get("parity_violation_rows") or 0),
        parity_violations=tuple(dict(item) for item in metadata.get("parity_violations", [])),
        no_arbitrage_checks=tuple(str(item) for item in metadata.get("no_arbitrage_checks", [])),
        no_arbitrage_violation_count=int(metadata.get("no_arbitrage_violation_count") or 0),
        no_arbitrage_violation_rows=int(metadata.get("no_arbitrage_violation_rows") or 0),
        no_arbitrage_reason_buckets=tuple(
            (str(reason), int(count))
            for reason, count in (metadata.get("no_arbitrage_reason_buckets") or {}).items()
        ),
        no_arbitrage_violations=tuple(dict(item) for item in metadata.get("no_arbitrage_violations", [])),
        no_arbitrage_excluded_count=int(metadata.get("no_arbitrage_excluded_count") or 0),
        raw_rows=int(metadata.get("raw_rows") or 0),
        valid_rows=int(metadata.get("valid_rows") or len(quotes)),
        rejected_rows=int(metadata.get("rejected_rows") or 0),
        warnings=tuple(metadata.get("warnings") or ()),
    )


def list_snapshots(symbol: str | None = None, directory: Path | str = DEFAULT_SNAPSHOT_DIR) -> list[Path]:
    """Return persisted snapshot metadata files, newest first."""
    root = Path(directory)
    if not root.exists():
        return []
    pattern = f"{symbol.upper()}_*.metadata.json" if symbol else "*.metadata.json"
    return sorted(root.glob(pattern), reverse=True)


def load_latest_snapshot(symbol: str, directory: Path | str = DEFAULT_SNAPSHOT_DIR) -> MarketDataSnapshot | None:
    """Load the newest persisted snapshot for ``symbol`` if one exists."""
    matches = list_snapshots(symbol, directory)
    return load_snapshot(matches[0]) if matches else None


def load_recent_snapshots(
    symbol: str,
    directory: Path | str = DEFAULT_SNAPSHOT_DIR,
    *,
    before: datetime | None = None,
    max_count: int | None = None,
) -> list[MarketDataSnapshot]:
    """Load recent persisted snapshots for ``symbol``, newest first."""
    snapshots: list[MarketDataSnapshot] = []
    for metadata_path in list_snapshots(symbol, directory):
        try:
            snapshot = load_snapshot(metadata_path)
        except Exception:
            continue
        if before is not None and snapshot.spot_timestamp >= before:
            continue
        snapshots.append(snapshot)
        if max_count is not None and len(snapshots) >= max_count:
            break
    return snapshots


def _snapshot_metadata(snapshot: MarketDataSnapshot) -> dict[str, Any]:
    options_name = f"{snapshot.symbol}_{snapshot.spot_timestamp.strftime('%Y%m%d_%H%M%S')}.options.parquet"
    return {
        "symbol": snapshot.symbol,
        "spot": snapshot.spot,
        "spot_timestamp": snapshot.spot_timestamp.isoformat(),
        "chain_timestamp": snapshot.chain_timestamp.isoformat() if snapshot.chain_timestamp else None,
        "source": snapshot.source,
        "source_delay_seconds": int(snapshot.source_delay.total_seconds()) if snapshot.source_delay else None,
        "cache_age_seconds": int(snapshot.cache_age.total_seconds()) if snapshot.cache_age else None,
        "fallback_reason": snapshot.fallback_reason,
        "mode": snapshot.mode,
        "risk_free_rate_source": snapshot.risk_free_rate_source,
        "risk_free_rate_mode": snapshot.risk_free_rate_mode,
        "risk_free_rate_timestamp": (
            snapshot.risk_free_rate_timestamp.isoformat() if snapshot.risk_free_rate_timestamp else None
        ),
        "risk_free_rate_fallback_reason": snapshot.risk_free_rate_fallback_reason,
        "risk_free_rate_curve": [
            {"tenor_days": tenor_days, "rate": rate} for tenor_days, rate in snapshot.risk_free_rate_curve
        ],
        "expiry_rates": dict(snapshot.expiry_rates),
        "risk_free_rate_30d": snapshot.risk_free_rate_30d,
        "risk_free_rate_min": snapshot.risk_free_rate_min,
        "risk_free_rate_max": snapshot.risk_free_rate_max,
        "risk_free_rate_median": snapshot.risk_free_rate_median,
        "dividend_source": snapshot.dividend_source,
        "dividend_mode": snapshot.dividend_mode,
        "dividend_timestamp": snapshot.dividend_timestamp.isoformat() if snapshot.dividend_timestamp else None,
        "dividend_fallback_reason": snapshot.dividend_fallback_reason,
        "annual_dividend_yield": snapshot.annual_dividend_yield,
        "dividend_events": list(snapshot.dividend_events),
        "expiry_dividends": dict(snapshot.expiry_dividends),
        "effective_dividend_yield_30d": snapshot.effective_dividend_yield_30d,
        "effective_dividend_yield_min": snapshot.effective_dividend_yield_min,
        "effective_dividend_yield_max": snapshot.effective_dividend_yield_max,
        "effective_dividend_yield_median": snapshot.effective_dividend_yield_median,
        "corporate_action_source": snapshot.corporate_action_source,
        "corporate_action_mode": snapshot.corporate_action_mode,
        "corporate_action_timestamp": (
            snapshot.corporate_action_timestamp.isoformat() if snapshot.corporate_action_timestamp else None
        ),
        "corporate_action_fallback_reason": snapshot.corporate_action_fallback_reason,
        "corporate_actions": list(snapshot.corporate_actions),
        "upcoming_corporate_actions": list(snapshot.upcoming_corporate_actions),
        "expiry_corporate_actions": dict(snapshot.expiry_corporate_actions),
        "corporate_action_warning_count": snapshot.corporate_action_warning_count,
        "corporate_action_warnings": list(snapshot.corporate_action_warnings),
        "stale_quote_count": snapshot.stale_quote_count,
        "last_only_quote_count": snapshot.last_only_quote_count,
        "zero_bid_ask_count": snapshot.zero_bid_ask_count,
        "crossed_market_count": snapshot.crossed_market_count,
        "locked_market_count": snapshot.locked_market_count,
        "crossed_locked_rejected_count": snapshot.crossed_locked_rejected_count,
        "stale_last_only_rejected_count": snapshot.stale_last_only_rejected_count,
        "min_open_interest": snapshot.min_open_interest,
        "min_volume": snapshot.min_volume,
        "max_bid_ask_spread_pct": snapshot.max_bid_ask_spread_pct,
        "liquidity_filtered_count": snapshot.liquidity_filtered_count,
        "low_open_interest_rejected_count": snapshot.low_open_interest_rejected_count,
        "low_volume_rejected_count": snapshot.low_volume_rejected_count,
        "wide_spread_rejected_count": snapshot.wide_spread_rejected_count,
        "old_quote_rejected_count": snapshot.old_quote_rejected_count,
        "rejection_reasons": dict(snapshot.rejection_reasons),
        "data_quality_score": snapshot.data_quality_score,
        "quality_score": snapshot.quality_score,
        "quality_reason_buckets": dict(snapshot.quality_reason_buckets),
        "expiry_quality": dict(snapshot.expiry_quality),
        "quote_reliability_summary": dict(snapshot.quote_reliability_summary),
        "fit_filters": dict(snapshot.fit_filters),
        "fit_filter_preset": snapshot.fit_filter_preset,
        "fit_eligible_count": snapshot.fit_eligible_count,
        "fit_excluded_count": snapshot.fit_excluded_count,
        "display_eligible_count": snapshot.display_eligible_count,
        "display_excluded_count": snapshot.display_excluded_count,
        "display_rejection_reason_buckets": dict(snapshot.display_rejection_reason_buckets),
        "fit_penalty_reason_buckets": dict(snapshot.fit_penalty_reason_buckets),
        "fit_hard_rejection_reason_buckets": dict(snapshot.fit_hard_rejection_reason_buckets),
        "expiry_reliability": dict(snapshot.expiry_reliability),
        "max_quote_age_days": snapshot.max_quote_age_days,
        "option_price_source": snapshot.option_price_source,
        "pricing_model": snapshot.pricing_model,
        "pricing_model_label": snapshot.pricing_model_label,
        "pricing_model_assumptions": snapshot.pricing_model_assumptions,
        "pricing_model_warning": snapshot.pricing_model_warning,
        "contract_greeks_count": snapshot.contract_greeks_count,
        "second_order_greeks_count": snapshot.second_order_greeks_count,
        "greek_units": dict(snapshot.greek_units),
        "median_selected_model_residual": snapshot.median_selected_model_residual,
        "max_abs_selected_model_residual": snapshot.max_abs_selected_model_residual,
        "computed_iv_count": snapshot.computed_iv_count,
        "computed_iv_failed_count": snapshot.computed_iv_failed_count,
        "parity_pairs_checked": snapshot.parity_pairs_checked,
        "parity_violation_count": snapshot.parity_violation_count,
        "parity_violation_rows": snapshot.parity_violation_rows,
        "parity_violations": list(snapshot.parity_violations),
        "no_arbitrage_checks": list(snapshot.no_arbitrage_checks),
        "no_arbitrage_violation_count": snapshot.no_arbitrage_violation_count,
        "no_arbitrage_violation_rows": snapshot.no_arbitrage_violation_rows,
        "no_arbitrage_reason_buckets": dict(snapshot.no_arbitrage_reason_buckets),
        "no_arbitrage_violations": list(snapshot.no_arbitrage_violations),
        "no_arbitrage_excluded_count": snapshot.no_arbitrage_excluded_count,
        "raw_rows": snapshot.raw_rows,
        "valid_rows": snapshot.valid_rows,
        "rejected_rows": snapshot.rejected_rows,
        "warnings": list(snapshot.warnings),
        "options_path": options_name,
    }


def _datetime_from_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _timedelta_from_seconds(value: int | float | None) -> timedelta | None:
    return timedelta(seconds=float(value)) if value is not None else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if value is not None else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None
