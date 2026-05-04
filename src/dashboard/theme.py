"""Dashboard styling and Plotly theme helpers."""

from __future__ import annotations

import plotly.graph_objects as go


CSS = """
<style>
    :root {
        --bg: #f6f7f9;
        --panel: #ffffff;
        --ink: #17202a;
        --muted: #667085;
        --line: #d9dee7;
        --accent: #1f7a8c;
        --good: #0f8a5f;
        --warn: #b7791f;
        --bad: #b42318;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        max-width: 1500px;
    }
    .workstation-header {
        border: 1px solid var(--line);
        background: var(--panel);
        padding: 0.9rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .workstation-title {
        font-size: 1.35rem;
        font-weight: 750;
        color: var(--ink);
        line-height: 1.2;
    }
    .workstation-subtitle {
        color: var(--muted);
        font-size: 0.86rem;
        margin-top: 0.2rem;
    }
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: var(--ink);
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.45rem;
        margin: 1rem 0 0.7rem 0;
    }
    .status-pill {
        display: inline-block;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.18rem 0.55rem;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.35rem;
        background: #f8fafc;
    }
    .status-live { color: var(--good); border-color: #b7e2d3; background: #edfdf7; }
    .status-synthetic { color: var(--warn); border-color: #f1d09a; background: #fff8e8; }
    .status-fallback { color: var(--bad); border-color: #f3b7b0; background: #fff1ef; }
    .quality-row {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.8rem;
        background: var(--panel);
        color: var(--muted);
        font-size: 0.85rem;
    }
    .loading-panel, .empty-panel {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        padding: 0.85rem 0.95rem;
        margin: 0.45rem 0 0.7rem 0;
    }
    .loading-panel-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }
    .loading-stage {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0;
        text-transform: uppercase;
    }
    .loading-title, .empty-title {
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 750;
        line-height: 1.25;
    }
    .loading-detail, .empty-detail {
        color: var(--muted);
        font-size: 0.82rem;
        margin-top: 0.15rem;
    }
    .loading-pulse {
        border: 1px solid #b7d7df;
        border-radius: 999px;
        color: var(--accent);
        background: #edf8fa;
        font-size: 0.68rem;
        font-weight: 800;
        padding: 0.15rem 0.45rem;
    }
    .skeleton-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.45rem;
        margin-top: 0.8rem;
    }
    .skeleton-line {
        height: 0.82rem;
        border-radius: 5px;
        background: linear-gradient(90deg, #eef1f5 25%, #f8fafc 45%, #eef1f5 65%);
        background-size: 220% 100%;
        animation: skeleton-shimmer 1.15s ease-in-out infinite;
    }
    .skeleton-line-1 { grid-column: span 1; }
    .skeleton-line-2 { grid-column: span 2; }
    .skeleton-line-3 { grid-column: span 3; }
    .skeleton-line-4 { grid-column: span 4; }
    .empty-panel {
        border-style: dashed;
        color: var(--muted);
    }
    .empty-action {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        margin-top: 0.45rem;
    }
    @keyframes skeleton-shimmer {
        0% { background-position: 120% 0; }
        100% { background-position: -120% 0; }
    }
    @media (max-width: 1024px) {
        .block-container {
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }
        .workstation-header {
            padding: 0.75rem 0.8rem;
        }
        .workstation-title {
            font-size: 1.12rem;
        }
        .workstation-subtitle, .quality-row, .small-note {
            font-size: 0.78rem;
        }
        .skeleton-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 640px) {
        .block-container {
            padding-left: 0.55rem;
            padding-right: 0.55rem;
            padding-top: 0.75rem;
        }
        .workstation-header, .quality-row, .loading-panel, .empty-panel {
            border-radius: 6px;
        }
        .workstation-title {
            font-size: 1rem;
        }
        .status-pill {
            margin-top: 0.25rem;
            padding: 0.16rem 0.42rem;
            font-size: 0.72rem;
        }
        .loading-panel-top {
            display: block;
        }
        .loading-pulse {
            display: inline-block;
            margin-top: 0.45rem;
        }
        .skeleton-grid {
            grid-template-columns: 1fr;
        }
    }
    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.8rem;
        background: var(--panel);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-size: 0.78rem;
    }
    .small-note {
        color: var(--muted);
        font-size: 0.82rem;
    }
</style>
"""


def inject_theme(st_module) -> None:
    st_module.markdown(CSS, unsafe_allow_html=True)


def data_mode_class(mode: str) -> str:
    lowered = (mode or "").lower()
    if "synthetic" in lowered:
        return "status-synthetic"
    if "fallback" in lowered or "unavailable" in lowered:
        return "status-fallback"
    return "status-live"


def status_pill(label: str, mode: str) -> str:
    return f'<span class="status-pill {data_mode_class(mode)}">{label}: {mode or "Unknown"}</span>'


def apply_chart_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=35, r=25, t=55, b=35),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color="#17202a"),
        hoverlabel=dict(bgcolor="#17202a", font_color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf0f5", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf0f5", zeroline=False)
    return fig
