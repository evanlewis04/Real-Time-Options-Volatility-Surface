"""Phase 6 dashboard component registry."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from html import escape
from typing import Iterator


@dataclass(frozen=True)
class DashboardComponentSpec:
    """Named UI component and the provenance fields it must keep visible."""

    key: str
    title: str
    workflow: str
    required_views: tuple[str, ...]
    provenance_fields: tuple[str, ...]


SurfaceWorkspace = DashboardComponentSpec(
    key="surface_workspace",
    title="SurfaceWorkspace",
    workflow="surface_analysis",
    required_views=(
        "3d_surface",
        "heatmap",
        "raw_points",
        "fit_residuals",
        "axis_controls",
        "fit_mode_controls",
        "quote_reliability_overlay",
    ),
    provenance_fields=(
        "surface_mode",
        "surface_source",
        "surface_quality_score",
        "pricing_model_label",
        "surface_estimate_type",
    ),
)

ChainExplorer = DashboardComponentSpec(
    key="chain_explorer",
    title="ChainExplorer",
    workflow="chain_explorer",
    required_views=("chain_grid", "filters", "iv_greeks", "liquidity_flags", "row_details"),
    provenance_fields=("source", "mode", "liquidity_filtered_count", "option_price_source"),
)

SkewLab = DashboardComponentSpec(
    key="skew_lab",
    title="SkewLab",
    workflow="skew_metrics",
    required_views=("smile_by_expiry", "delta_skew", "risk_reversal", "butterfly", "raw_vs_fitted"),
    provenance_fields=("surface_source", "surface_mode", "fit_diagnostics"),
)

TermStructurePanel = DashboardComponentSpec(
    key="term_structure_panel",
    title="TermStructurePanel",
    workflow="term_structure",
    required_views=("atm_iv_curve", "realized_vol_overlay", "event_markers", "front_back_spread"),
    provenance_fields=("source", "surface_mode", "event_source"),
)

DataQualityPanel = DashboardComponentSpec(
    key="data_quality_panel",
    title="DataQualityPanel",
    workflow="data_quality",
    required_views=(
        "source",
        "timestamp",
        "cache_age",
        "rejected_rows",
        "violations",
        "fit_errors",
        "quality_drop_alert",
        "actionable_reasons",
    ),
    provenance_fields=("source", "timestamp", "cache_age_seconds", "quality_reason_buckets", "quality_drop_alert"),
)

ScannerPanel = DashboardComponentSpec(
    key="scanner_panel",
    title="ScannerPanel",
    workflow="cross_symbol_scanner",
    required_views=("iv_rank", "skew", "term_slope", "rich_cheap_residuals"),
    provenance_fields=("source", "mode", "surface_source"),
)

StrategyBuilder = DashboardComponentSpec(
    key="strategy_builder",
    title="StrategyBuilder",
    workflow="strategy_pricing",
    required_views=("leg_editor", "payoff_chart", "greeks_table", "scenario_controls", "surface_pricing"),
    provenance_fields=("source", "pricing_model_label", "surface_source"),
)

PortfolioRiskPanel = DashboardComponentSpec(
    key="portfolio_risk_panel",
    title="PortfolioRiskPanel",
    workflow="portfolio_risk",
    required_views=("position_import", "aggregate_greeks", "scenario_pnl", "concentration", "hedges"),
    provenance_fields=("source", "mode", "pricing_model_label"),
)

DiagnosticsPanel = DashboardComponentSpec(
    key="diagnostics_panel",
    title="DiagnosticsPanel",
    workflow="provenance_health",
    required_views=("provider_health", "latency", "exceptions", "capabilities", "latest_logs"),
    provenance_fields=("overall", "data_contract", "performance"),
)

ReportExportPanel = DashboardComponentSpec(
    key="report_export_panel",
    title="ReportExportPanel",
    workflow="report_export",
    required_views=("html_export", "notebook_export", "workspace_export", "fit_diagnostics_export"),
    provenance_fields=("source", "mode", "data_timestamp", "model_assumptions"),
)


PHASE6_COMPONENTS: tuple[DashboardComponentSpec, ...] = (
    SurfaceWorkspace,
    ChainExplorer,
    SkewLab,
    TermStructurePanel,
    DataQualityPanel,
    ScannerPanel,
    StrategyBuilder,
    PortfolioRiskPanel,
    DiagnosticsPanel,
    ReportExportPanel,
)


def phase6_component_titles() -> list[str]:
    """Return Phase 6 component titles in implementation order."""
    return [component.title for component in PHASE6_COMPONENTS]


@contextmanager
def card(
    st_module,
    *,
    title: str,
    kicker: str | None = None,
    actions: list[str] | None = None,
) -> Iterator[None]:
    """Render the shared workstation card chrome around Streamlit content."""
    action_markup = "".join(
        f'<span class="panel-card-action" title="{escape(action)}">{escape(action)}</span>'
        for action in (actions or [])
    )
    kicker_markup = f'<div class="panel-card-kicker">{escape(kicker)}</div>' if kicker else ""
    st_module.markdown(
        f"""
        <section class="panel-card">
            <header class="panel-card-header">
                <div>
                    {kicker_markup}
                    <div class="panel-card-title">{escape(title)}</div>
                </div>
                <div class="panel-card-actions">{action_markup}</div>
            </header>
            <div class="panel-card-body">
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        st_module.markdown("</div></section>", unsafe_allow_html=True)
