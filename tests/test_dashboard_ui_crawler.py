from scripts import dashboard_ui_crawler as crawler
from src.dashboard.app_shell import (
    coerce_table_numeric_columns,
    display_provenance_label,
    format_scanner_table_for_display,
)


def test_crawler_tab_specs_cover_dashboard_tabs():
    assert list(crawler.TAB_SPECS) == [
        "F1 Surface",
        "F2 Chain",
        "F3 Skew",
        "F4 Term",
        "F5 Quality",
        "F6 Scanner",
        "F7 Strategy",
        "F8 Risk",
        "F9 Diag",
        "F10 Export",
    ]

    assert "Data Quality Panel" in crawler.TAB_SPECS["F5 Quality"]
    assert "Report Export Panel" in crawler.TAB_SPECS["F10 Export"]


def test_crawler_slug_makes_stable_artifact_names():
    assert crawler._slug("Report Export Panel") == "report_export_panel"
    assert crawler._slug("AAPL/MSFT: Surface") == "aapl_msft_surface"


def test_crawler_detects_literal_dashboard_html_fragments():
    assert crawler.HTML_FRAGMENT_RE.search('<div class="metric-card">')
    assert crawler.HTML_FRAGMENT_RE.search('data-dashboard-section="kpi-grid"')
    assert not crawler.HTML_FRAGMENT_RE.search("Options Volatility Surface Workstation")


def test_crawler_detects_truncated_kpi_values():
    assert crawler.TRUNCATED_KPI_RE.match("$...")
    assert crawler.TRUNCATED_KPI_RE.match("2...")
    assert not crawler.TRUNCATED_KPI_RE.match("$196.50")
    assert not crawler.TRUNCATED_KPI_RE.match("n/a")


def test_crawler_default_output_dir_tracks_data_mode():
    assert crawler._default_output_dir("offline") == "artifacts/dashboard_crawler"
    assert crawler._default_output_dir("online") == "artifacts/dashboard_crawler_online"


def test_crawler_offline_env_forces_deterministic_connector(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "external::test")
    monkeypatch.setenv("VOL_SURFACE_APPTEST_MODE", "provider_failure")

    env = crawler._streamlit_env("offline")

    assert env["PYTEST_CURRENT_TEST"] == "scripts.dashboard_ui_crawler::offline_browser_crawl"
    assert env["VOL_SURFACE_APPTEST_MODE"] == "synthetic"


def test_crawler_online_env_removes_offline_test_hooks(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "external::test")
    monkeypatch.setenv("VOL_SURFACE_APPTEST_MODE", "synthetic")

    env = crawler._streamlit_env("online")

    assert "PYTEST_CURRENT_TEST" not in env
    assert "VOL_SURFACE_APPTEST_MODE" not in env


def test_dashboard_provenance_label_is_human_readable():
    assert display_provenance_label("prior_assisted_fit_estimate_not_market_observation") == (
        "Prior Assisted Fit Estimate"
    )
    assert display_provenance_label("raw_quote_diagnostic_overlay") == "Raw Quote Diagnostic Overlay"


def test_dashboard_table_numeric_coercion_enables_column_formatting():
    import pandas as pd

    frame = pd.DataFrame({"market_iv": ["0.2456789"], "classification": ["rich"]})

    coerced = coerce_table_numeric_columns(frame, ["market_iv"])

    assert coerced["market_iv"].iloc[0] == 0.2456789
    assert coerced["market_iv"].dtype.kind == "f"


def test_dashboard_scanner_table_display_limits_float_precision():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "market_iv": ["2.0651696459871625"],
            "fitted_iv": [1.6191320834941083],
            "surface_residual": [0.44603756249305415],
            "bid_ask_spread_pct": [0.02710027100271],
            "residual_z_score": [5.789123],
            "liquidity_score": [0.70456],
            "strike": [520.0],
            "dte": [1.0],
        }
    )

    display = format_scanner_table_for_display(frame)

    assert display["market_iv"].iloc[0] == "206.52%"
    assert display["fitted_iv"].iloc[0] == "161.91%"
    assert display["surface_residual"].iloc[0] == "44.60%"
    assert display["bid_ask_spread_pct"].iloc[0] == "2.71%"
    assert display["residual_z_score"].iloc[0] == "5.79"
    assert display["liquidity_score"].iloc[0] == "0.70"
    assert display["strike"].iloc[0] == "$520.00"
    assert display["dte"].iloc[0] == "1"
