"""Performance timing utilities for provider and calculation steps."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Iterator


@dataclass(frozen=True)
class PerformanceRecord:
    """One timed operation."""

    operation: str
    latency_ms: float
    timestamp: datetime
    symbol: str | None = None
    provider: str | None = None
    source: str | None = None
    cache_hit: bool | None = None
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerformanceRecorder:
    """Bounded in-memory recorder for recent timings."""

    def __init__(self, max_records: int = 200):
        self.max_records = max(1, int(max_records))
        self.records: list[PerformanceRecord] = []

    @contextmanager
    def measure(
        self,
        operation: str,
        *,
        symbol: str | None = None,
        provider: str | None = None,
        source: str | None = None,
        cache_hit: bool | None = None,
        fallback_reason: str | None = None,
    ) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self.record(
                operation,
                (perf_counter() - start) * 1000.0,
                symbol=symbol,
                provider=provider,
                source=source,
                cache_hit=cache_hit,
                fallback_reason=fallback_reason,
            )

    def record(
        self,
        operation: str,
        latency_ms: float,
        *,
        symbol: str | None = None,
        provider: str | None = None,
        source: str | None = None,
        cache_hit: bool | None = None,
        fallback_reason: str | None = None,
    ) -> PerformanceRecord:
        record = PerformanceRecord(
            operation=operation,
            latency_ms=round(float(latency_ms), 3),
            timestamp=datetime.now(),
            symbol=symbol.upper() if symbol else None,
            provider=provider,
            source=source,
            cache_hit=cache_hit,
            fallback_reason=fallback_reason,
        )
        self.records.append(record)
        if len(self.records) > self.max_records:
            del self.records[: len(self.records) - self.max_records]
        return record

    def slowest(self, limit: int = 8) -> list[dict[str, Any]]:
        """Return the slowest recent records as dictionaries."""
        ordered = sorted(self.records, key=lambda item: item.latency_ms, reverse=True)
        return [record.as_dict() for record in ordered[: max(0, int(limit))]]

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the newest recent records as dictionaries."""
        return [record.as_dict() for record in self.records[-max(0, int(limit)):]]
