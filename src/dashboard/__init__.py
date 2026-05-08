"""Streamlit dashboard package."""

from .app_shell import run_dashboard
from .components import PHASE6_COMPONENTS, DashboardComponentSpec, phase6_component_titles
from .pages import PageSpec, default_page_registry, page_by_key, page_titles
from .state import DashboardStateService

__all__ = [
    "DashboardStateService",
    "DashboardComponentSpec",
    "PHASE6_COMPONENTS",
    "PageSpec",
    "default_page_registry",
    "page_by_key",
    "page_titles",
    "phase6_component_titles",
    "run_dashboard",
]
