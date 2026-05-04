"""Loading and empty-state helpers for Streamlit dashboard panels."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class LoadingState:
    """Description for a temporary loading panel."""

    title: str
    detail: str
    stage: str
    rows: int = 3


def render_loading_state(state: LoadingState) -> str:
    """Return dense skeleton markup for a slow data fetch."""
    rows = max(1, min(int(state.rows), 8))
    bars = "\n".join(
        f'<div class="skeleton-line skeleton-line-{(index % 4) + 1}"></div>'
        for index in range(rows)
    )
    return f"""
    <div class="loading-panel" role="status" aria-live="polite">
        <div class="loading-panel-top">
            <div>
                <div class="loading-stage">{escape(state.stage)}</div>
                <div class="loading-title">{escape(state.title)}</div>
                <div class="loading-detail">{escape(state.detail)}</div>
            </div>
            <div class="loading-pulse">FETCH</div>
        </div>
        <div class="skeleton-grid">
            {bars}
        </div>
    </div>
    """


def render_empty_state(title: str, detail: str, action: str | None = None) -> str:
    """Return compact empty-state markup with an optional recovery action."""
    action_markup = f'<div class="empty-action">{escape(action)}</div>' if action else ""
    return f"""
    <div class="empty-panel">
        <div class="empty-title">{escape(title)}</div>
        <div class="empty-detail">{escape(detail)}</div>
        {action_markup}
    </div>
    """


def load_with_status(st_module, state: LoadingState, loader: Callable[[], T]) -> T:
    """Render a skeleton placeholder while ``loader`` performs a blocking fetch."""
    slot = st_module.empty()
    slot.markdown(render_loading_state(state), unsafe_allow_html=True)
    try:
        return loader()
    finally:
        slot.empty()
