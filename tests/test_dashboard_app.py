from streamlit.testing.v1 import AppTest

from scripts.dashboard_visual_regression import VIEWPORTS


def test_dashboard_default_state_renders_key_sections():
    at = AppTest.from_file("app.py")
    at.run(timeout=90)

    assert not at.exception
    assert len(at.metric) == 8
    assert len(at.tabs) == 6
    assert len(at.dataframe) >= 2


def test_visual_regression_viewports_cover_desktop_tablet_mobile():
    assert VIEWPORTS["desktop"] == (1440, 1000)
    assert VIEWPORTS["tablet"][0] == 1024
    assert VIEWPORTS["mobile"][0] < 640
