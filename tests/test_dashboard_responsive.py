import tomllib
from pathlib import Path

import plotly.graph_objects as go

from src.dashboard.theme import CSS, apply_chart_layout


def test_theme_has_tablet_and_mobile_breakpoints():
    assert "@media (max-width: 1024px)" in CSS
    assert "@media (max-width: 640px)" in CSS


def test_mobile_breakpoint_stacks_loading_layout():
    mobile_css = CSS.split("@media (max-width: 640px)", 1)[1]

    assert ".loading-panel-top" in mobile_css
    assert "display: block" in mobile_css
    assert "grid-template-columns: 1fr" in mobile_css


def test_theme_polishes_workstation_controls_without_large_radius_cards():
    assert 'section[data-testid="stSidebar"]' in CSS
    assert 'div[data-testid="stTabs"] button[aria-selected="true"]' in CSS
    assert 'div[data-testid="stDataFrame"], div[data-testid="stTable"]' in CSS
    assert ".metric-grid" in CSS
    assert ".quality-group-grid" in CSS
    assert ".dashboard-ready-marker" in CSS
    assert "border-radius: 8px" in CSS


def test_kpi_grid_uses_two_rows_on_desktop():
    assert "grid-template-columns: repeat(5, minmax(0, 1fr))" in CSS
    assert "overflow-wrap: anywhere" in CSS


def test_streamlit_metric_values_do_not_ellipsis_truncate():
    assert 'div[data-testid="stMetricValue"]' in CSS
    assert "text-overflow: clip" in CSS
    assert "white-space: normal" in CSS


def test_theme_styles_finance_grade_controls_and_tabs():
    assert 'div[data-baseweb="select"] > div' in CSS
    assert 'span[data-baseweb="tag"]' in CSS
    assert 'div[data-testid="stSlider"] [role="slider"]' in CSS
    assert 'div[data-testid="stTabs"] button[aria-label]' in CSS
    assert 'div[data-baseweb="popover"] [role="listbox"]' in CSS
    assert "var(--focus)" in CSS


def test_plotly_3d_scene_uses_dark_workstation_theme():
    fig = go.Figure(data=[go.Surface(z=[[0.2, 0.3], [0.25, 0.35]], colorbar=dict(title="IV"))])
    fig.update_layout(scene=dict(xaxis_title="Moneyness", yaxis_title="Days", zaxis_title="IV"))

    themed = apply_chart_layout(fig, 620)

    assert themed.layout.paper_bgcolor == "#181b20"
    assert themed.layout.scene.xaxis.backgroundcolor == "#14171b"
    assert themed.layout.scene.yaxis.gridcolor == "#2a3038"
    assert themed.data[0].colorbar.bgcolor == "#14171b"


def test_streamlit_native_components_use_dark_theme():
    config_path = Path(".streamlit/config.toml")
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["theme"]["base"] == "dark"
    assert config["theme"]["secondaryBackgroundColor"] == "#181b20"
    assert config["theme"]["primaryColor"] == "#d89a2b"
