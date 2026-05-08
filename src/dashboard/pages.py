"""Page registry for independently loaded dashboard workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.dashboard.components import PHASE6_COMPONENTS
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
        PageSpec(component.key, component.title, component.workflow, _page_payload)
        for component in PHASE6_COMPONENTS
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
