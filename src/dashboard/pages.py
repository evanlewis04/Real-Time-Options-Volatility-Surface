"""Page registry for independently loaded dashboard workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.dashboard.state import DashboardStateService


PageRenderer = Callable[[DashboardStateService, Any], Any]


@dataclass(frozen=True)
class PageSpec:
    """Metadata and loader for one dashboard page."""

    key: str
    title: str
    workflow: str
    render: PageRenderer


def default_page_registry() -> list[PageSpec]:
    """Return the app's independent workflow pages in display order."""
    return [
        PageSpec("surface", "Surface", "surface_analysis", _page_payload),
        PageSpec("chain", "Chain", "chain_explorer", _page_payload),
        PageSpec("skew_term", "Skew & Term", "skew_term_structure", _page_payload),
        PageSpec("local_vol", "Local Vol", "local_volatility", _page_payload),
        PageSpec("relative_value", "Relative Value", "cross_symbol_scanner", _page_payload),
        PageSpec("strategy_lab", "Strategy Lab", "strategy_pricing", _page_payload),
        PageSpec("risk", "Risk", "portfolio_risk", _page_payload),
        PageSpec("diagnostics", "Diagnostics", "provenance_health", _page_payload),
    ]


def page_titles(registry: list[PageSpec] | None = None) -> list[str]:
    return [page.title for page in (registry or default_page_registry())]


def page_by_key(key: str, registry: list[PageSpec] | None = None) -> PageSpec | None:
    for page in registry or default_page_registry():
        if page.key == key:
            return page
    return None


def _page_payload(state: DashboardStateService, connector: Any) -> dict[str, Any]:
    return {
        "state": state.snapshot(),
        "connector": connector.__class__.__name__ if connector is not None else None,
    }
