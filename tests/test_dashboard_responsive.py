from src.dashboard.theme import CSS


def test_theme_has_tablet_and_mobile_breakpoints():
    assert "@media (max-width: 1024px)" in CSS
    assert "@media (max-width: 640px)" in CSS


def test_mobile_breakpoint_stacks_loading_layout():
    mobile_css = CSS.split("@media (max-width: 640px)", 1)[1]

    assert ".loading-panel-top" in mobile_css
    assert "display: block" in mobile_css
    assert "grid-template-columns: 1fr" in mobile_css
