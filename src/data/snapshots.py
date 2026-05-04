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
