from scripts import dashboard_ui_crawler as crawler


def test_crawler_tab_specs_cover_dashboard_tabs():
    assert list(crawler.TAB_SPECS) == [
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

    assert "Data Quality Panel" in crawler.TAB_SPECS["DataQualityPanel"]
    assert "Report Export Panel" in crawler.TAB_SPECS["ReportExportPanel"]


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
