"""Shared dashboard state and data-service helpers for multi-page views."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable


@dataclass
class DashboardStateService:
    """Central state/data cache shared by independent dashboard pages."""

    selected_symbol: str = "AAPL"
    selected_symbols: list[str] = field(default_factory=lambda: ["AAPL"])
    data_key: tuple[Any, ...] = field(default_factory=tuple)
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _data: dict[str, Any] = field(default_factory=dict)

    def set_context(
        self,
        *,
        selected_symbol: str | None = None,
        selected_symbols: list[str] | tuple[str, ...] | None = None,
        data_key: tuple[Any, ...] | None = None,
    ) -> None:
        if selected_symbol:
            self.selected_symbol = selected_symbol.upper()
        if selected_symbols is not None:
            self.selected_symbols = [str(symbol).upper() for symbol in selected_symbols]
        if data_key is not None:
            self.data_key = tuple(data_key)
        self.updated_at = datetime.now(UTC)

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Any:
        if key not in self._data:
            self._data[key] = loader()
            self.updated_at = datetime.now(UTC)
        return self._data[key]

    def put(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.updated_at = datetime.now(UTC)

    def invalidate(self, prefix: str | None = None) -> None:
        if prefix is None:
            self._data.clear()
        else:
            for key in [item for item in self._data if item.startswith(prefix)]:
                self._data.pop(key, None)
        self.updated_at = datetime.now(UTC)

    def snapshot(self) -> dict[str, Any]:
        return {
            "selected_symbol": self.selected_symbol,
            "selected_symbols": list(self.selected_symbols),
            "data_key": list(self.data_key),
            "updated_at": self.updated_at.isoformat(),
            "cached_keys": sorted(self._data),
        }
