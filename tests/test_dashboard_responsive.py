from src.dashboard.theme import CSS


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
