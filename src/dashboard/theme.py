"""Dashboard styling and Plotly theme helpers."""

from __future__ import annotations

import plotly.graph_objects as go


CSS = """
<style>
    :root {
        --bg: #111316;
        --panel: #181b20;
        --panel-2: #20242a;
        --ink: #f2f5f8;
        --muted: #9aa4af;
        --line: #343a43;
        --accent: #d89a2b;
        --accent-2: #20a4a8;
        --good: #4dbd74;
        --warn: #e4a83b;
        --bad: #ff6b5f;
        --focus: #f2b84b;
    }
    .stApp {
        background: var(--bg);
        color: var(--ink);
    }
    header[data-testid="stHeader"] {
        background: var(--bg);
        height: 0;
    }
    div[data-testid="stToolbar"], #MainMenu, footer {
        visibility: hidden;
        height: 0;
    }
    .block-container {
        padding-top: 0.85rem;
        padding-bottom: 1.5rem;
        max-width: 1500px;
    }
    .workstation-header {
        border: 1px solid var(--line);
        background: #14171b;
        padding: 0.75rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    .workstation-topline {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
    }
    .workstation-kicker {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .workstation-title {
        font-size: 1.22rem;
        font-weight: 800;
        color: var(--ink);
        line-height: 1.2;
    }
    .workstation-subtitle, .workstation-tape {
        color: var(--muted);
        font-size: 0.78rem;
    }
    .workstation-symbol-block {
        text-align: right;
        min-width: 9rem;
    }
    .workstation-symbol {
        color: var(--accent);
        font-size: 1.05rem;
        font-weight: 850;
    }
    .workstation-spot {
        color: var(--ink);
        font-size: 0.95rem;
        font-weight: 750;
    }
    .workstation-tape {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.35rem;
        margin-top: 0.55rem;
    }
    .workstation-tape span {
        border: 1px solid #2a3038;
        background: #101215;
        border-radius: 6px;
        padding: 0.35rem 0.45rem;
        overflow-wrap: anywhere;
    }
    .workstation-tape strong {
        color: var(--ink);
    }
    .workstation-readiness {
        border-left: 3px solid var(--accent);
        background: #101215;
        margin-top: 0.55rem;
        padding: 0.45rem 0.6rem;
        border-radius: 6px;
    }
    .readiness-title {
        color: var(--accent);
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .readiness-detail {
        color: var(--ink);
        font-size: 0.8rem;
        line-height: 1.35;
    }
    .status-rail {
        margin-top: 0.55rem;
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
        border-radius: 6px;
        padding: 0.18rem 0.55rem;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
        background: #101215;
    }
    .status-live { color: var(--good); border-color: rgba(77, 189, 116, 0.55); background: #102018; }
    .status-synthetic { color: var(--warn); border-color: rgba(228, 168, 59, 0.6); background: #251d0c; }
    .status-fallback { color: var(--bad); border-color: rgba(255, 107, 95, 0.55); background: #281413; }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.55rem;
        margin-bottom: 0.65rem;
    }
    .metric-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.62rem 0.7rem;
        background: var(--panel);
        min-width: 0;
    }
    .metric-card-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .metric-card-value {
        color: var(--ink);
        font-size: 1.08rem;
        font-weight: 850;
        line-height: 1.2;
        margin-top: 0.12rem;
        overflow-wrap: anywhere;
    }
    .metric-card-detail {
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 700;
        margin-top: 0.12rem;
        overflow-wrap: anywhere;
    }
    .quality-row, .quality-workstation {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.7rem;
        background: var(--panel);
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 0.65rem;
    }
    .quality-alert {
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.45rem 0.55rem;
        margin-bottom: 0.5rem;
        color: var(--ink);
        background: #101215;
    }
    .quality-alert-warning { border-color: rgba(228, 168, 59, 0.65); color: var(--warn); }
    .quality-alert-success { border-color: rgba(77, 189, 116, 0.55); color: var(--good); }
    .quality-alert-info { border-color: rgba(32, 164, 168, 0.55); color: #79d5d8; }
    .quality-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-bottom: 0.55rem;
    }
    .quality-chip {
        border: 1px solid #424953;
        border-radius: 6px;
        background: #101215;
        color: var(--muted);
        padding: 0.2rem 0.42rem;
        font-size: 0.72rem;
    }
    .quality-chip strong {
        color: var(--accent);
    }
    .quality-chip-muted {
        color: var(--good);
    }
    .quality-group-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.55rem;
    }
    .quality-group {
        border: 1px solid #2e343d;
        border-radius: 8px;
        background: #14171b;
        padding: 0.55rem;
        min-width: 0;
    }
    .quality-group-title {
        color: var(--accent);
        font-size: 0.74rem;
        font-weight: 850;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .quality-items {
        display: grid;
        gap: 0.3rem;
    }
    .quality-item {
        display: grid;
        grid-template-columns: minmax(7rem, 0.75fr) minmax(0, 1.2fr) minmax(5rem, 0.7fr);
        gap: 0.45rem;
        align-items: baseline;
        border-top: 1px solid #252a31;
        padding-top: 0.28rem;
    }
    .quality-item-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 750;
    }
    .quality-item-value {
        color: var(--ink);
        font-size: 0.78rem;
        font-weight: 750;
        overflow-wrap: anywhere;
    }
    .quality-item-note {
        color: var(--muted);
        font-size: 0.7rem;
        overflow-wrap: anywhere;
    }
    .dashboard-ready-marker {
        position: absolute;
        width: 1px;
        height: 1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
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
        border: 1px solid rgba(216, 154, 43, 0.6);
        border-radius: 999px;
        color: var(--accent);
        background: #251d0c;
        font-size: 0.68rem;
        font-weight: 800;
        padding: 0.15rem 0.45rem;
    }
    .loading-progress {
        height: 0.42rem;
        border: 1px solid #343a43;
        border-radius: 999px;
        background: #101215;
        margin-top: 0.65rem;
        overflow: hidden;
    }
    .loading-progress-fill {
        height: 100%;
        background: var(--accent);
        border-radius: 999px;
    }
    .loading-progress-text {
        color: var(--muted);
        font-size: 0.7rem;
        margin-top: 0.22rem;
        text-align: right;
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
        background: linear-gradient(90deg, #242a31 25%, #323943 45%, #242a31 65%);
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
        .workstation-tape, .metric-grid, .quality-group-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .quality-item {
            grid-template-columns: minmax(6rem, 0.9fr) minmax(0, 1.1fr);
        }
        .quality-item-note {
            grid-column: 2;
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
        .workstation-topline {
            display: block;
        }
        .workstation-symbol-block {
            text-align: left;
            margin-top: 0.35rem;
        }
        .workstation-tape, .metric-grid, .quality-group-grid {
            grid-template-columns: 1fr;
        }
        .quality-item {
            grid-template-columns: 1fr;
            gap: 0.1rem;
        }
        .quality-item-note {
            grid-column: auto;
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
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
        background: #14171b;
    }
    section[data-testid="stSidebar"] h3 {
        color: var(--ink);
        font-size: 0.92rem;
        margin-top: 0.7rem;
    }
    div[data-testid="stTabs"] button {
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 0.8rem;
        color: var(--muted);
        font-weight: 650;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--focus);
        border-bottom-color: var(--focus);
    }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        overflow: hidden;
        background: var(--panel);
    }
    div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button {
        border-radius: 6px;
        border-color: #b9c4d3;
        font-weight: 700;
    }
    div[data-testid="stDownloadButton"] button:hover, div[data-testid="stButton"] button:hover {
        border-color: var(--focus);
        color: var(--focus);
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
        template="plotly_dark",
        height=height,
        margin=dict(l=35, r=25, t=55, b=35),
        paper_bgcolor="#181b20",
        plot_bgcolor="#14171b",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color="#f2f5f8"),
        hoverlabel=dict(bgcolor="#f2f5f8", font_color="#111316"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a3038", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#2a3038", zeroline=False)
    return fig
