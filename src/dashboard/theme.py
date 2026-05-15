"""Dashboard styling and Plotly theme helpers.

Design token map:
- Color tokens are semantic: surfaces, borders, text, accent, data states, and
  series colors. Components should consume tokens instead of raw hex values.
- Type tokens define one UI family and one numeric mono family. Numeric UI uses
  tabular numbers and a slashed zero for scan-heavy workstation reading.
- Space tokens follow a 4-point scale. Component gaps, padding, and margins use
  these values so dense layouts stay regular across breakpoints.
- Radius tokens stop at 8px. Panels use hairline borders plus a 1px inset
  highlight instead of heavy shadows.
- Motion tokens are limited to subtle 120ms/200ms hover and focus treatments.
"""

from __future__ import annotations

import plotly.graph_objects as go


SURFACE_0 = "#0B0D10"
SURFACE_1 = "#11141A"
SURFACE_2 = "#171B22"
SURFACE_3 = "#1E232B"
BORDER_SUBTLE = "rgba(255, 255, 255, 0.06)"
BORDER_DEFAULT = "rgba(255, 255, 255, 0.12)"
BORDER_STRONG = "rgba(255, 255, 255, 0.2)"
TEXT_PRIMARY = "#F4F7FB"
TEXT_SECONDARY = "#B6C0CC"
TEXT_TERTIARY = "#7F8A98"
ACCENT = "#F5A524"
DATA_LIVE = "#22D3A8"
DATA_SYNTHETIC = "#F5A524"
DATA_FALLBACK = "#FF5C5C"
DATA_STALE = "#A79D90"

CHART_BG = SURFACE_0
PANEL_BG = SURFACE_1
GRID_COLOR = "rgba(255,255,255,0.08)"
LINE_COLOR = "rgba(255,255,255,0.16)"
INK = TEXT_PRIMARY
MUTED = TEXT_SECONDARY

SERIES_PALETTE = (
    "#22D3A8",
    "#60A5FA",
    "#F5A524",
    "#F472B6",
    "#A78BFA",
    "#34D399",
    "#FB7185",
    "#FACC15",
)

DIVERGING_SCALE = [
    [0.00, "#1C2A5E"],
    [0.18, "#2563EB"],
    [0.36, "#22D3A8"],
    [0.50, "#E7ECEF"],
    [0.64, "#F5A524"],
    [0.82, "#F97316"],
    [1.00, "#B42318"],
]

SEQUENTIAL_SCALE = [
    [0.00, "#0B0D10"],
    [0.18, "#132331"],
    [0.36, "#14545D"],
    [0.56, "#168F83"],
    [0.76, "#F5A524"],
    [1.00, "#FFE08A"],
]


CSS = """
<style>
    /* === tokens === */
    :root {
        --surface-0: #0B0D10;
        --surface-1: #11141A;
        --surface-2: #171B22;
        --surface-3: #1E232B;
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-default: rgba(255, 255, 255, 0.12);
        --border-strong: rgba(255, 255, 255, 0.2);
        --text-primary: #F4F7FB;
        --text-secondary: #B6C0CC;
        --text-tertiary: #7F8A98;
        --text-inverse: #0B0D10;
        --accent: #F5A524;
        --accent-hover: #FFC85A;
        --accent-muted: rgba(245, 165, 36, 0.08);
        --data-live: #22D3A8;
        --data-synthetic: #F5A524;
        --data-fallback: #FF5C5C;
        --data-stale: #A79D90;
        --series-1: #22D3A8;
        --series-2: #60A5FA;
        --series-3: #F5A524;
        --series-4: #F472B6;
        --series-5: #A78BFA;
        --series-6: #34D399;
        --series-7: #FB7185;
        --series-8: #FACC15;
        --font-ui: "Inter", "Segoe UI", Arial, sans-serif;
        --font-mono: "JetBrains Mono", "Cascadia Mono", "IBM Plex Mono", Consolas, monospace;
        --text-xs: 11px;
        --text-sm: 12px;
        --text-base: 13px;
        --text-md: 14px;
        --text-lg: 16px;
        --text-xl: 20px;
        --text-display: 28px;
        --s-1: 4px;
        --s-2: 8px;
        --s-3: 12px;
        --s-4: 16px;
        --s-5: 20px;
        --s-6: 24px;
        --s-7: 28px;
        --s-8: 32px;
        --r-sm: 4px;
        --r-md: 6px;
        --r-lg: 8px;
        --ease-out: cubic-bezier(0.2, 0.8, 0.2, 1);
        --motion-standard: 120ms;
        --motion-emphasis: 200ms;
        --panel-highlight: inset 0 1px 0 rgba(255, 255, 255, 0.03);

        /* Compatibility aliases for existing selectors and tests. */
        --bg: var(--surface-0);
        --panel: var(--surface-1);
        --panel-2: var(--surface-2);
        --ink: var(--text-primary);
        --muted: var(--text-secondary);
        --line: var(--border-default);
        --focus: var(--accent);
        --good: var(--data-live);
        --warn: var(--data-synthetic);
        --bad: var(--data-fallback);
    }

    /* === reset === */
    .stApp {
        background: var(--surface-0);
        color: var(--text-primary);
        font-family: var(--font-ui);
        font-size: var(--text-base);
    }
    .stApp * {
        letter-spacing: 0;
    }
    .stApp :is(code, pre, [data-testid="stMetricValue"], .mono, .metric-card-value, .workstation-symbol, .workstation-spot, .workstation-clock, .status-pill, .kpi-strip-value, .rail-context-spot, .rail-chip, .workstation-tape strong),
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum", "zero";
    }
    header[data-testid="stHeader"] {
        background: var(--surface-0);
        height: 0;
    }
    div[data-testid="stToolbar"], #MainMenu, footer {
        visibility: hidden;
        height: 0;
    }
    .block-container {
        padding: var(--s-3) var(--s-5) var(--s-6);
        max-width: 1540px;
    }
    a, a:visited { color: var(--accent); }
    :focus-visible {
        outline: 2px solid var(--accent-hover) !important;
        outline-offset: 2px !important;
        border-radius: var(--r-sm);
    }

    /* === topbar === */
    .workstation-header {
        border: 1px solid var(--border-default);
        background: var(--surface-1);
        border-radius: var(--r-lg);
        margin-bottom: var(--s-3);
        box-shadow: var(--panel-highlight);
        overflow: hidden;
    }
    .workstation-topline {
        min-height: 32px;
        display: flex;
        justify-content: space-between;
        gap: var(--s-4);
        align-items: center;
        padding: var(--s-2) var(--s-3);
        border-bottom: 1px solid var(--border-subtle);
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
    }
    .brand-cluster, .header-cluster {
        display: flex;
        align-items: center;
        gap: var(--s-2);
        min-width: 0;
    }
    .brand-mark {
        color: var(--accent);
        font-family: var(--font-mono);
        font-size: var(--text-md);
        font-weight: 800;
    }
    .workstation-kicker {
        color: var(--accent);
        font-size: var(--text-xs);
        font-weight: 800;
        line-height: 1;
        text-transform: uppercase;
    }
    .workstation-title {
        font-size: var(--text-xl);
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .workstation-subtitle, .workstation-tape {
        color: var(--text-secondary);
        font-size: var(--text-sm);
    }
    .env-tag, .shortcut-key {
        border: 1px solid var(--border-default);
        border-radius: var(--r-sm);
        background: var(--surface-2);
        color: var(--text-secondary);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        font-weight: 700;
        line-height: 1;
        padding: 4px 6px;
        text-transform: uppercase;
    }
    .workstation-clock {
        color: var(--text-secondary);
        font-size: var(--text-xs);
        font-weight: 700;
    }
    .ticker-strip {
        display: grid;
        grid-template-columns: minmax(180px, 1.15fr) repeat(5, minmax(92px, 0.7fr));
        align-items: stretch;
        min-height: 44px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .workstation-symbol-block {
        display: flex;
        align-items: baseline;
        gap: var(--s-3);
        padding: var(--s-2) var(--s-3);
        min-width: 0;
    }
    .workstation-symbol {
        color: var(--text-primary);
        font-size: var(--text-display);
        font-weight: 800;
        line-height: 1;
    }
    .workstation-spot {
        color: var(--accent);
        font-size: var(--text-lg);
        font-weight: 800;
    }
    .spot-delta {
        color: var(--data-live);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        font-weight: 700;
    }
    .kpi-strip-tile {
        border-left: 1px solid var(--border-subtle);
        padding: var(--s-2) var(--s-3);
        min-width: 0;
    }
    .kpi-strip-label {
        color: var(--text-tertiary);
        font-size: var(--text-xs);
        font-weight: 800;
        line-height: 1.1;
        text-transform: uppercase;
    }
    .kpi-strip-value {
        color: var(--text-primary);
        font-size: var(--text-md);
        font-weight: 800;
        margin-top: 2px;
        overflow-wrap: anywhere;
    }
    .workstation-tape {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        margin: 0;
        border-bottom: 1px solid var(--border-subtle);
    }
    .workstation-tape span {
        border-right: 1px solid var(--border-subtle);
        background: var(--surface-1);
        padding: var(--s-2) var(--s-3);
        overflow-wrap: anywhere;
    }
    .workstation-tape strong {
        color: var(--text-primary);
        font-weight: 750;
    }
    .workstation-readiness {
        border-left: 2px solid var(--accent);
        background: var(--accent-muted);
        margin: var(--s-3);
        padding: var(--s-2) var(--s-3);
        border-radius: var(--r-md);
    }
    .readiness-title {
        color: var(--accent);
        font-size: var(--text-xs);
        font-weight: 800;
        text-transform: uppercase;
    }
    .readiness-detail {
        color: var(--text-primary);
        font-size: var(--text-sm);
        line-height: 1.35;
    }
    .status-rail {
        display: flex;
        flex-wrap: wrap;
        gap: var(--s-1);
        padding: 0 var(--s-3) var(--s-3);
    }

    /* === command rail === */
    section[data-testid="stSidebar"] {
        width: 280px !important;
        border-right: 1px solid var(--border-default);
        background: var(--surface-1);
        box-shadow: var(--panel-highlight);
    }
    section[data-testid="stSidebar"] > div {
        padding: var(--s-3) var(--s-3) var(--s-5);
    }
    .rail-heading {
        color: var(--accent);
        font-size: var(--text-xs);
        font-weight: 800;
        text-transform: uppercase;
        margin: var(--s-3) 0 var(--s-2);
    }
    .rail-panel {
        border: 1px solid var(--border-default);
        border-radius: var(--r-md);
        background: var(--surface-2);
        box-shadow: var(--panel-highlight);
        padding: var(--s-3);
        margin-bottom: var(--s-3);
    }
    .rail-command-label {
        color: var(--text-tertiary);
        font-size: var(--text-xs);
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: var(--s-1);
    }
    .rail-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: var(--s-1);
        margin: var(--s-2) 0 var(--s-3);
    }
    .rail-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        height: 24px;
        border: 1px solid var(--border-default);
        border-radius: var(--r-sm);
        background: var(--surface-0);
        color: var(--text-secondary);
        font-size: var(--text-xs);
        font-weight: 700;
        padding: 0 var(--s-2);
    }
    .rail-chip.active {
        border-color: rgba(245,165,36,0.55);
        background: var(--accent-muted);
        color: var(--text-primary);
    }
    .rail-dot, .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: var(--data-live);
        flex: 0 0 auto;
    }
    .rail-dot.synthetic, .status-dot.synthetic { background: var(--data-synthetic); }
    .rail-dot.fallback, .status-dot.fallback { background: var(--data-fallback); }
    .rail-dot.stale, .status-dot.stale { background: var(--data-stale); }
    .rail-context {
        display: grid;
        gap: var(--s-1);
    }
    .rail-context-symbol {
        color: var(--text-primary);
        font-family: var(--font-mono);
        font-size: var(--text-xl);
        font-weight: 800;
        line-height: 1;
    }
    .rail-context-spot {
        color: var(--accent);
        font-size: var(--text-lg);
        font-weight: 800;
    }
    .rail-context-meta {
        color: var(--text-tertiary);
        font-size: var(--text-xs);
    }
    .rail-footer {
        color: var(--text-tertiary);
        font-size: var(--text-xs);
        line-height: 1.5;
        border-top: 1px solid var(--border-subtle);
        margin-top: var(--s-4);
        padding-top: var(--s-3);
    }
    .rail-footer strong {
        color: var(--text-secondary);
        font-family: var(--font-mono);
    }
    section[data-testid="stSidebar"] details[data-testid="stExpander"] {
        border: 1px solid var(--border-default);
        border-radius: var(--r-md);
        background: var(--surface-2);
        box-shadow: var(--panel-highlight);
        margin-bottom: var(--s-3);
        overflow: hidden;
    }
    section[data-testid="stSidebar"] details[data-testid="stExpander"] summary {
        min-height: 34px;
        color: var(--text-primary);
        font-size: var(--text-xs);
        font-weight: 800;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border-subtle);
        background: var(--surface-1);
        padding: 0 var(--s-2);
    }
    section[data-testid="stSidebar"] details[data-testid="stExpander"] summary:hover {
        background: var(--surface-3);
        color: var(--accent-hover);
    }
    section[data-testid="stSidebar"] details[data-testid="stExpander"] > div {
        padding: var(--s-2) var(--s-2) var(--s-3);
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        min-height: 26px;
        border-radius: var(--r-sm);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        padding: 0 var(--s-2);
        text-transform: uppercase;
    }

    /* === native Streamlit controls === */
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: var(--text-secondary);
        font-size: var(--text-sm);
    }
    div[data-testid="stTextInput"] input,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background: var(--surface-0);
        border: 1px solid var(--border-default);
        border-radius: var(--r-md);
        color: var(--text-primary);
        min-height: 34px;
        box-shadow: none;
        transition: border-color var(--motion-standard) var(--ease-out), background var(--motion-standard) var(--ease-out);
    }
    div[data-testid="stTextInput"] input:hover,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:hover,
    div[data-testid="stNumberInput"] input:hover {
        border-color: rgba(245,165,36,0.5);
        background: var(--surface-2);
    }
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg {
        color: var(--text-primary);
        fill: var(--text-primary);
    }
    div[data-baseweb="popover"] [role="listbox"],
    ul[data-testid="stVirtualDropdown"] {
        background: var(--surface-1);
        border: 1px solid var(--border-default);
        border-radius: var(--r-md);
        box-shadow: 0 16px 38px rgba(0, 0, 0, 0.45);
    }
    div[data-baseweb="popover"] [role="option"] {
        color: var(--text-primary);
        font-weight: 650;
    }
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [aria-selected="true"] {
        background: var(--accent-muted);
        color: var(--accent-hover);
    }
    span[data-baseweb="tag"] {
        background: rgba(34, 211, 168, 0.08);
        border: 1px solid rgba(34, 211, 168, 0.38);
        border-radius: var(--r-sm);
        color: var(--text-primary);
        font-weight: 700;
    }
    span[data-baseweb="tag"] span { color: var(--text-primary); }
    span[data-baseweb="tag"] svg { fill: var(--text-tertiary); }
    div[data-testid="stSlider"] {
        padding-top: 0;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: 4px;
        padding-bottom: 4px;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] div {
        border-radius: 999px;
    }
    div[data-testid="stSlider"] [role="slider"] {
        width: 10px;
        height: 10px;
        background: var(--accent);
        border-color: var(--accent);
        box-shadow: 0 0 0 2px rgba(245, 165, 36, 0.18);
    }
    div[data-testid="stCheckbox"] label {
        color: var(--text-primary);
    }
    div[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p,
    div[data-testid="stSlider"] [data-testid="stWidgetLabel"] p,
    div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p,
    div[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p {
        color: var(--text-secondary);
        font-size: var(--text-sm);
        font-weight: 650;
    }
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stButton"] button {
        min-height: 34px;
        border-radius: var(--r-md);
        border: 1px solid var(--border-default);
        background: var(--surface-2);
        color: var(--text-primary);
        font-weight: 750;
        transition: all var(--motion-standard) var(--ease-out);
    }
    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stButton"] button:hover,
    div[data-testid="stDownloadButton"] button:hover {
        border-color: rgba(245,165,36,0.55);
        background: var(--accent-muted);
        color: var(--accent-hover);
    }

    /* === tabs === */
    div[data-testid="stTabs"] [role="tablist"] {
        min-height: 36px;
        border: 1px solid var(--border-default);
        border-radius: var(--r-md);
        background: var(--surface-1);
        box-shadow: var(--panel-highlight);
        gap: 0;
        padding: 0 var(--s-1);
        overflow-x: auto;
    }
    div[data-testid="stTabs"] button {
        background: transparent;
        border: 0;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        color: var(--text-tertiary);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        font-weight: 750;
        min-height: 36px;
        padding: 0 var(--s-3);
        text-transform: uppercase;
        white-space: nowrap;
    }
    div[data-testid="stTabs"] button:hover {
        background: var(--surface-2);
        color: var(--text-primary);
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--accent);
        background: transparent;
        border-bottom-color: var(--accent);
    }
    div[data-testid="stTabs"] button[aria-label],
    div[data-testid="stTabs"] [role="tablist"] ~ button {
        background: var(--surface-2);
        border: 1px solid var(--border-default);
        border-radius: var(--r-md);
        color: var(--accent);
        box-shadow: none;
        min-width: 2rem;
    }
    div[data-testid="stTabs"] button[aria-label] svg,
    div[data-testid="stTabs"] [role="tablist"] ~ button svg {
        fill: var(--accent);
        color: var(--accent);
    }

    /* === cards and sections === */
    .section-header {
        display: flex;
        align-items: center;
        min-height: 34px;
        color: var(--text-primary);
        font-size: var(--text-md);
        font-weight: 750;
        border: 1px solid var(--border-default);
        border-radius: var(--r-md) var(--r-md) 0 0;
        background: var(--surface-1);
        box-shadow: var(--panel-highlight);
        padding: 0 var(--s-3);
        margin: var(--s-4) 0 0;
    }
    .panel-card {
        border: 1px solid var(--border-default);
        border-radius: var(--r-lg);
        background: var(--surface-1);
        box-shadow: var(--panel-highlight);
        margin-bottom: var(--s-3);
        overflow: hidden;
    }
    .panel-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--s-3);
        padding: var(--s-2) var(--s-3);
        border-bottom: 1px solid var(--border-subtle);
    }
    .panel-card-kicker {
        color: var(--accent);
        font-size: var(--text-xs);
        font-weight: 800;
        text-transform: uppercase;
        line-height: 1.1;
    }
    .panel-card-title {
        color: var(--text-primary);
        font-size: var(--text-md);
        font-weight: 750;
        line-height: 1.2;
    }
    .panel-card-actions {
        display: flex;
        align-items: center;
        gap: var(--s-1);
        color: var(--text-tertiary);
    }
    .panel-card-action {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 24px;
        height: 24px;
        border: 1px solid var(--border-default);
        border-radius: var(--r-sm);
        background: var(--surface-2);
        color: var(--text-tertiary);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        font-weight: 800;
        padding: 0 var(--s-1);
        text-transform: uppercase;
    }
    .panel-card-action:hover {
        border-color: rgba(245,165,36,0.55);
        color: var(--accent-hover);
        background: var(--accent-muted);
    }
    .panel-card-body {
        padding: var(--s-3);
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0;
        margin-bottom: var(--s-3);
        border: 1px solid var(--border-default);
        border-radius: var(--r-lg);
        background: var(--surface-1);
        box-shadow: var(--panel-highlight);
        overflow: hidden;
    }
    .metric-card {
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        padding: var(--s-3);
        background: var(--surface-1);
        min-width: 0;
    }
    .metric-card-label {
        color: var(--text-tertiary);
        font-size: var(--text-xs);
        font-weight: 800;
        text-transform: uppercase;
    }
    .metric-card-value {
        color: var(--text-primary);
        font-size: var(--text-lg);
        font-weight: 850;
        line-height: 1.2;
        margin-top: var(--s-1);
        overflow-wrap: anywhere;
    }
    .metric-card-detail {
        color: var(--accent);
        font-size: var(--text-xs);
        font-weight: 700;
        margin-top: 2px;
        overflow-wrap: anywhere;
    }
    .quality-row, .quality-workstation {
        border: 1px solid var(--border-default);
        border-radius: var(--r-lg);
        padding: var(--s-3);
        background: var(--surface-1);
        box-shadow: var(--panel-highlight);
        color: var(--text-secondary);
        font-size: var(--text-sm);
        margin-bottom: var(--s-3);
    }
    .quality-alert {
        display: flex;
        align-items: center;
        gap: var(--s-2);
        border: 1px solid var(--border-default);
        border-radius: var(--r-md);
        padding: var(--s-2) var(--s-3);
        margin-bottom: var(--s-3);
        color: var(--text-primary);
        background: var(--surface-2);
    }
    .quality-alert::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: var(--data-stale);
    }
    .quality-alert-warning { border-color: rgba(245, 165, 36, 0.55); }
    .quality-alert-warning::before { background: var(--data-synthetic); }
    .quality-alert-success { border-color: rgba(34, 211, 168, 0.45); }
    .quality-alert-success::before { background: var(--data-live); }
    .quality-alert-info { border-color: rgba(96, 165, 250, 0.45); }
    .quality-alert-info::before { background: #60A5FA; }
    .quality-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: var(--s-1);
        margin-bottom: var(--s-3);
    }
    .quality-chip {
        border: 1px solid var(--border-default);
        border-radius: var(--r-sm);
        background: var(--surface-0);
        color: var(--text-secondary);
        padding: 3px var(--s-2);
        font-size: var(--text-xs);
    }
    .quality-chip strong { color: var(--accent); }
    .quality-chip-muted { color: var(--data-live); }
    .quality-group-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: var(--s-3);
    }
    .quality-group {
        border: 1px solid var(--border-subtle);
        border-radius: var(--r-md);
        background: var(--surface-2);
        box-shadow: var(--panel-highlight);
        padding: var(--s-3);
        min-width: 0;
    }
    .quality-group-title {
        color: var(--accent);
        font-size: var(--text-xs);
        font-weight: 850;
        text-transform: uppercase;
        margin-bottom: var(--s-2);
    }
    .quality-items {
        display: grid;
        gap: var(--s-1);
    }
    .quality-item {
        display: grid;
        grid-template-columns: minmax(7rem, 0.75fr) minmax(0, 1.2fr) minmax(5rem, 0.7fr);
        gap: var(--s-2);
        align-items: baseline;
        border-top: 1px solid var(--border-subtle);
        padding-top: var(--s-1);
    }
    .quality-item-label {
        color: var(--text-tertiary);
        font-size: var(--text-xs);
        font-weight: 750;
    }
    .quality-item-value {
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-weight: 750;
        overflow-wrap: break-word;
        word-break: normal;
    }
    .quality-item-note {
        color: var(--text-tertiary);
        font-size: var(--text-xs);
        overflow-wrap: anywhere;
    }

    /* === status and states === */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        height: 20px;
        border: 1px solid var(--border-default);
        border-radius: var(--r-sm);
        padding: 0 var(--s-2);
        font-size: var(--text-xs);
        font-weight: 800;
        text-transform: uppercase;
        background: var(--surface-2);
        color: var(--text-secondary);
    }
    .status-pill::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: var(--data-live);
    }
    .status-live { color: var(--data-live); border-color: rgba(34, 211, 168, 0.45); background: rgba(34, 211, 168, 0.08); }
    .status-synthetic { color: var(--data-synthetic); border-color: rgba(245, 165, 36, 0.5); background: var(--accent-muted); }
    .status-fallback { color: var(--data-fallback); border-color: rgba(255, 92, 92, 0.45); background: rgba(255, 92, 92, 0.08); }
    .status-synthetic::before { background: var(--data-synthetic); }
    .status-fallback::before { background: var(--data-fallback); }
    .dashboard-ready-marker {
        position: absolute;
        width: 1px;
        height: 1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
    }
    .loading-panel, .empty-panel {
        border: 1px solid var(--border-default);
        border-radius: var(--r-lg);
        background: var(--surface-1);
        box-shadow: var(--panel-highlight);
        padding: var(--s-4);
        margin: var(--s-2) 0 var(--s-3);
    }
    .loading-panel-top, .empty-panel-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: var(--s-3);
    }
    .state-icon {
        width: 24px;
        height: 24px;
        color: var(--accent);
        flex: 0 0 auto;
    }
    .loading-copy, .empty-copy { min-width: 0; }
    .loading-stage {
        color: var(--accent);
        font-size: var(--text-xs);
        font-weight: 800;
        text-transform: uppercase;
    }
    .loading-title, .empty-title {
        color: var(--text-primary);
        font-size: var(--text-md);
        font-weight: 750;
        line-height: 1.25;
    }
    .loading-detail, .empty-detail {
        color: var(--text-secondary);
        font-size: var(--text-sm);
        margin-top: 2px;
    }
    .loading-pulse {
        border: 1px solid rgba(245, 165, 36, 0.5);
        border-radius: var(--r-sm);
        color: var(--accent);
        background: var(--accent-muted);
        font-family: var(--font-mono);
        font-size: var(--text-xs);
        font-weight: 800;
        padding: 3px var(--s-2);
    }
    .loading-progress {
        height: 5px;
        border: 1px solid var(--border-subtle);
        border-radius: 999px;
        background: var(--surface-0);
        margin-top: var(--s-3);
        overflow: hidden;
    }
    .loading-progress-fill {
        height: 100%;
        background: var(--accent);
        border-radius: 999px;
    }
    .loading-progress-text {
        color: var(--text-tertiary);
        font-size: var(--text-xs);
        margin-top: var(--s-1);
        text-align: right;
    }
    .skeleton-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: var(--s-2);
        margin-top: var(--s-3);
    }
    .skeleton-line {
        height: 12px;
        border-radius: var(--r-sm);
        background: linear-gradient(90deg, #1B2028 25%, #242A33 45%, #1B2028 65%);
        background-size: 220% 100%;
        animation: skeleton-shimmer 1.15s ease-in-out infinite;
    }
    .skeleton-line-1 { grid-column: span 1; }
    .skeleton-line-2 { grid-column: span 2; }
    .skeleton-line-3 { grid-column: span 3; }
    .skeleton-line-4 { grid-column: span 4; }
    .empty-panel {
        border-style: dashed;
        color: var(--text-secondary);
    }
    .empty-action {
        display: inline-flex;
        align-items: center;
        min-height: 28px;
        border: 1px solid rgba(245,165,36,0.45);
        border-radius: var(--r-md);
        color: var(--accent);
        background: var(--accent-muted);
        font-size: var(--text-sm);
        font-weight: 750;
        margin-top: var(--s-3);
        padding: 0 var(--s-3);
    }
    @keyframes skeleton-shimmer {
        0% { background-position: 120% 0; }
        100% { background-position: -120% 0; }
    }

    /* === tables === */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid var(--border-default);
        border-radius: var(--r-lg);
        overflow: hidden;
        background: var(--surface-1);
        box-shadow: var(--panel-highlight);
    }
    div[data-testid="stDataFrame"] * {
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum", "zero";
    }
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stTable"] th {
        position: sticky;
        top: 0;
        background: var(--surface-2) !important;
        color: var(--text-secondary) !important;
        font-size: var(--text-xs) !important;
        font-weight: 800 !important;
        text-transform: uppercase;
    }
    div[data-testid="stDataFrame"] [role="row"],
    div[data-testid="stTable"] tr {
        min-height: 28px;
    }
    div[data-testid="stTable"] tbody tr:nth-child(even),
    div[data-testid="stDataFrame"] [role="row"]:nth-child(even) {
        background: rgba(255, 255, 255, 0.02);
    }

    /* === metrics compatibility === */
    div[data-testid="stMetric"] {
        border: 1px solid var(--border-default);
        border-radius: var(--r-lg);
        padding: var(--s-3);
        background: var(--surface-1);
        box-shadow: var(--panel-highlight);
    }
    div[data-testid="stMetricLabel"] p {
        color: var(--text-tertiary);
        font-size: var(--text-sm);
    }
    div[data-testid="stMetricValue"] {
        color: var(--text-primary);
        font-size: var(--text-lg) !important;
        line-height: 1.2;
        overflow-wrap: anywhere !important;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
    }
    div[data-testid="stMetricValue"] * {
        font-size: inherit !important;
        line-height: inherit !important;
    }
    .small-note {
        color: var(--text-secondary);
        font-size: var(--text-sm);
    }

    /* === responsive === */
    @media (max-width: 1024px) {
        .block-container {
            padding-left: var(--s-3);
            padding-right: var(--s-3);
        }
        .ticker-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .workstation-symbol-block {
            grid-column: span 2;
        }
        .workstation-title { font-size: var(--text-lg); }
        .workstation-subtitle, .quality-row, .small-note { font-size: var(--text-sm); }
        .workstation-tape, .metric-grid, .quality-group-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .quality-item {
            grid-template-columns: minmax(6rem, 0.9fr) minmax(0, 1.1fr);
        }
        .quality-item-note { grid-column: 2; }
        .skeleton-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
        .block-container {
            padding-left: var(--s-2);
            padding-right: var(--s-2);
            padding-top: var(--s-2);
        }
        .workstation-header, .quality-row, .loading-panel, .empty-panel {
            border-radius: var(--r-md);
        }
        .workstation-topline, .ticker-strip, .workstation-tape, .metric-grid, .quality-group-grid {
            display: grid;
            grid-template-columns: 1fr;
        }
        .header-cluster {
            justify-content: flex-start;
            flex-wrap: wrap;
        }
        .workstation-symbol-block {
            grid-column: auto;
            padding: var(--s-3);
        }
        .workstation-symbol { font-size: var(--text-xl); }
        .kpi-strip-tile {
            border-left: 0;
            border-top: 1px solid var(--border-subtle);
        }
        .quality-item {
            grid-template-columns: 1fr;
            gap: 2px;
        }
        .quality-item-note { grid-column: auto; }
        .status-pill {
            margin-top: var(--s-1);
            padding: 0 var(--s-2);
            font-size: var(--text-xs);
        }
        .loading-panel-top, .empty-panel-top {
            display: block;
        }
        .loading-pulse {
            display: inline-block;
            margin-top: var(--s-2);
        }
        .skeleton-grid { grid-template-columns: 1fr; }
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


def _dark_colorbar() -> dict:
    return {
        "bgcolor": PANEL_BG,
        "bordercolor": LINE_COLOR,
        "borderwidth": 1,
        "tickfont": {"color": TEXT_SECONDARY, "family": "JetBrains Mono, Consolas, monospace", "size": 11},
        "title": {"font": {"color": TEXT_SECONDARY, "size": 11}},
        "outlinecolor": LINE_COLOR,
    }


def _normalize_colorscale(value: object) -> list[list[object]] | object:
    if value is None:
        return value
    value_text = str(value).lower()
    diverging_names = ("rdbu", "rdylgn", "balance", "delta", "diverging")
    sequential_names = ("viridis", "plasma", "cividis", "inferno", "magma", "turbo")
    if any(name in value_text for name in diverging_names):
        return DIVERGING_SCALE
    if any(name in value_text for name in sequential_names):
        return SEQUENTIAL_SCALE
    return value


def _has_explicit_color(value: object) -> bool:
    """Return True when Plotly already has a scalar or array color configured."""
    if value is None:
        return False
    try:
        return len(value) > 0  # type: ignore[arg-type]
    except TypeError:
        return True


def _apply_trace_colorbars(fig: go.Figure) -> None:
    colorbar_style = _dark_colorbar()
    palette_index = 0
    for trace in fig.data:
        colorbar = getattr(trace, "colorbar", None)
        if colorbar is not None:
            colorbar.update(colorbar_style)
        marker = getattr(trace, "marker", None)
        marker_colorbar = getattr(marker, "colorbar", None)
        if marker_colorbar is not None:
            marker_colorbar.update(colorbar_style)
        if hasattr(trace, "colorscale") and getattr(trace, "colorscale", None) is not None:
            trace.update(colorscale=_normalize_colorscale(getattr(trace, "colorscale", None)))
        marker_colorscale = getattr(marker, "colorscale", None)
        if marker is not None and marker_colorscale is not None:
            marker.update(colorscale=_normalize_colorscale(marker_colorscale))
        if getattr(trace, "type", "") in {"scatter", "scatter3d"}:
            line = getattr(trace, "line", None)
            marker_obj = getattr(trace, "marker", None)
            series_color = SERIES_PALETTE[palette_index % len(SERIES_PALETTE)]
            if line is not None and not _has_explicit_color(getattr(line, "color", None)):
                line.update(color=series_color)
            if marker_obj is not None and not _has_explicit_color(getattr(marker_obj, "color", None)):
                marker_obj.update(color=series_color)
            palette_index += 1


def _dark_scene_axis(title: str | None = None) -> dict:
    axis = {
        "backgroundcolor": SURFACE_0,
        "gridcolor": GRID_COLOR,
        "zerolinecolor": LINE_COLOR,
        "showbackground": True,
        "tickfont": {"color": TEXT_SECONDARY, "family": "JetBrains Mono, Consolas, monospace", "size": 11},
        "title": {"font": {"color": TEXT_SECONDARY, "size": 11}},
    }
    if title:
        axis["title"]["text"] = title
    return axis


def apply_chart_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply the shared dark workstation theme to every Plotly figure."""
    _apply_trace_colorbars(fig)
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=35, r=25, t=48, b=35),
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color=TEXT_PRIMARY),
        title=dict(font=dict(size=13, color=TEXT_PRIMARY), x=0.01, xanchor="left"),
        hoverlabel=dict(
            bgcolor=SURFACE_1,
            bordercolor=ACCENT,
            font=dict(color=TEXT_PRIMARY, family="JetBrains Mono, Consolas, monospace", size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(color=TEXT_SECONDARY, size=11),
            itemsizing="constant",
        ),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=TEXT_TERTIARY, activecolor=ACCENT),
        colorway=list(SERIES_PALETTE),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=LINE_COLOR,
        tickfont=dict(color=TEXT_SECONDARY, family="JetBrains Mono, Consolas, monospace", size=11),
        title_font=dict(color=TEXT_SECONDARY, size=11),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
        linecolor=LINE_COLOR,
        tickfont=dict(color=TEXT_SECONDARY, family="JetBrains Mono, Consolas, monospace", size=11),
        title_font=dict(color=TEXT_SECONDARY, size=11),
    )
    if getattr(fig.layout, "scene", None):
        scene = fig.layout.scene
        fig.update_layout(
            scene={
                "bgcolor": SURFACE_1,
                "xaxis": _dark_scene_axis(getattr(scene.xaxis.title, "text", None)),
                "yaxis": _dark_scene_axis(getattr(scene.yaxis.title, "text", None)),
                "zaxis": _dark_scene_axis(getattr(scene.zaxis.title, "text", None)),
            }
        )
    return fig
