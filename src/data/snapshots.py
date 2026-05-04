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
        max_quote_age_days=_int_or_none(metadata.get("max_quote_age_days")),
        option_price_source=metadata.get("option_price_source", "mark"),
        computed_iv_count=int(metadata.get("computed_iv_count") or 0),
        computed_iv_failed_count=int(metadata.get("computed_iv_failed_count") or 0),
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
        "max_quote_age_days": snapshot.max_quote_age_days,
        "option_price_source": snapshot.option_price_source,
        "computed_iv_count": snapshot.computed_iv_count,
        "computed_iv_failed_count": snapshot.computed_iv_failed_count,
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
