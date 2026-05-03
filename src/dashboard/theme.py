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
