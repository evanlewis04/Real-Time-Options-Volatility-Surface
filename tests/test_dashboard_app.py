from streamlit.testing.v1 import AppTest
import streamlit as st

from scripts.dashboard_visual_regression import VIEWPORTS


def _run_app(monkeypatch, mode=None):
    if mode is None:
        monkeypatch.delenv("VOL_SURFACE_APPTEST_MODE", raising=False)
    else:
        monkeypatch.setenv("VOL_SURFACE_APPTEST_MODE", mode)
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file("app.py")
    at.run(timeout=90)
    return at


def test_dashboard_default_state_renders_key_sections():
    at = AppTest.from_file("app.py")
    at.run(timeout=90)

    assert not at.exception
    assert len(at.metric) >= 10
    assert any(metric.label == "ATM dIV" for metric in at.metric)
    assert any(metric.label == "Exp Move" for metric in at.metric)
    assert len(at.tabs) == 10
    assert [tab.label for tab in at.tabs] == [
        "SurfaceWorkspace",
        "ChainExplorer",
        "SkewLab",
        "TermStructurePanel",
        "DataQualityPanel",
        "ScannerPanel",
        "StrategyBuilder",
        "PortfolioRiskPanel",
        "DiagnosticsPanel",
        "ReportExportPanel",
    ]
    assert len(at.dataframe) >= 2


def test_dashboard_no_symbol_state_renders_without_exception(monkeypatch):
    at = _run_app(monkeypatch)
    universe = next(widget for widget in at.multiselect if widget.label == "Universe")

    universe.set_value([])
    at.run(timeout=90)

    assert not at.exception
    assert any("Select at least one symbol" in warning.value for warning in at.warning)


def test_dashboard_synthetic_mode_renders_with_visible_provenance(monkeypatch):
    at = _run_app(monkeypatch, "synthetic")

    assert not at.exception
    assert any("Surface: Synthetic" in item.value for item in at.markdown)
    assert any("Price: Synthetic" in item.value for item in at.markdown)


def test_dashboard_provider_failure_mode_renders_diagnostics(monkeypatch):
    at = _run_app(monkeypatch, "provider_failure")

    assert not at.exception
    assert any("Forced provider failure for deterministic AppTest coverage" in item.value for item in at.json)
    assert any("Surface: Fallback" in item.value for item in at.markdown)


def test_dashboard_fit_preset_selector_renders_without_exception(monkeypatch):
    at = _run_app(monkeypatch)
    preset = next(widget for widget in at.selectbox if widget.label == "Fit preset")

    preset.set_value("Strict")
    at.run(timeout=90)
    assert not at.exception

    preset = next(widget for widget in at.selectbox if widget.label == "Fit preset")
    preset.set_value("Diagnostic Raw")
    at.run(timeout=90)
    assert not at.exception


def test_visual_regression_viewports_cover_desktop_tablet_mobile():
    assert VIEWPORTS["desktop"] == (1440, 1000)
    assert VIEWPORTS["tablet"][0] == 1024
    assert VIEWPORTS["mobile"][0] < 640
