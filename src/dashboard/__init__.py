"""Streamlit dashboard package."""

from .app_shell import run_dashboard
from .pages import PageSpec, default_page_registry, page_by_key, page_titles
from .state import DashboardStateService

__all__ = [
    "DashboardStateService",
    "PageSpec",
    "default_page_registry",
    "page_by_key",
    "page_titles",
    "run_dashboard",
]
