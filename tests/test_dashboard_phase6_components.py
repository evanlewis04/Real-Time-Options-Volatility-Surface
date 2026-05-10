from src.dashboard.components import PHASE6_COMPONENTS, phase6_component_titles
from src.dashboard.pages import default_page_registry, page_titles


def test_phase6_component_registry_is_complete_and_ordered():
    assert phase6_component_titles() == [
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

    assert len(PHASE6_COMPONENTS) == 10
    assert PHASE6_COMPONENTS[0].required_views == (
        "3d_surface",
        "heatmap",
        "raw_points",
        "fit_residuals",
        "axis_controls",
        "fit_mode_controls",
        "quote_reliability_overlay",
    )
    assert "source" in PHASE6_COMPONENTS[4].provenance_fields
    assert "notebook_export" in PHASE6_COMPONENTS[-1].required_views
    assert "fit_diagnostics_export" in PHASE6_COMPONENTS[-1].required_views


def test_page_registry_exposes_phase6_components():
    registry = default_page_registry()

    assert page_titles(registry) == phase6_component_titles()
    assert [page.key for page in registry] == [component.key for component in PHASE6_COMPONENTS]
