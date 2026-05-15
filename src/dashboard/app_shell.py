from __future__ import annotations

from html import escape

import pandas as pd

from src.dashboard.formatting import fmt_int, fmt_money, fmt_pct
from src.quant.provenance import (
    CURRENT_ROBUST_FIT_PROVENANCE,
    ML_DENOISED_PROVENANCE,
    PRIOR_ASSISTED_FIT_PROVENANCE,
    RAW_QUOTE_DIAGNOSTIC_PROVENANCE,
    STANDARD_SVI_FIT_PROVENANCE,
)

FIT_MODE_CHOICES = ("Robust", "Standard", "Prior Assisted", "ML Denoised", "Diagnostic Raw")
FIT_PRESETS = {
    "Standard": {
        "max_bid_ask_spread_pct": 0.75,
        "max_quote_age_days": 5,
        "min_volume": 0,
        "min_open_interest": 0,
        "moneyness": (0.50, 2.00),
        "max_raw_iv": 2.00,
        "no_arbitrage_policy": "exclude",
        "last_only_policy": "allow_penalized",
    },
    "Strict": {
        "max_bid_ask_spread_pct": 0.35,
        "max_quote_age_days": 2,
        "min_volume": 10,
        "min_open_interest": 50,
        "moneyness": (0.70, 1.35),
        "max_raw_iv": 1.50,
        "no_arbitrage_policy": "exclude",
        "last_only_policy": "exclude",
    },
    "Diagnostic Raw": {
        "max_bid_ask_spread_pct": 1.50,
        "max_quote_age_days": 0,
        "min_volume": 0,
        "min_open_interest": 0,
        "moneyness": (0.35, 2.50),
        "max_raw_iv": 5.00,
        "no_arbitrage_policy": "allow",
        "last_only_policy": "allow",
    },
}
SCANNER_NUMERIC_COLUMNS = [
    "dte",
    "strike",
    "market_iv",
    "fitted_iv",
    "surface_residual",
    "residual_z_score",
    "liquidity_score",
    "bid_ask_spread_pct",
    "volume",
    "open_interest",
]
SCANNER_PERCENT_COLUMNS = ["market_iv", "fitted_iv", "surface_residual", "bid_ask_spread_pct"]
PROVENANCE_DISPLAY_LABELS = {
    CURRENT_ROBUST_FIT_PROVENANCE: "Robust Fit Estimate",
    STANDARD_SVI_FIT_PROVENANCE: "Standard SVI Fit Estimate",
    PRIOR_ASSISTED_FIT_PROVENANCE: "Prior Assisted Fit Estimate",
    ML_DENOISED_PROVENANCE: "ML Denoised Research Estimate",
    RAW_QUOTE_DIAGNOSTIC_PROVENANCE: "Raw Quote Diagnostic Overlay",
    "current_fit_estimate_not_market_observation": "Current Fit Estimate",
    "historical_prior_estimate_not_market_observation": "Historical Prior Estimate",
    "synthetic_surface_estimate": "Synthetic Surface Estimate",
}


def _html(value: object) -> str:
    """Escape a value for dashboard HTML fragments."""
    return escape("n/a" if value is None else str(value), quote=True)


def display_provenance_label(value: object) -> str:
    """Return human-readable provenance copy without changing underlying metadata."""
    raw = "" if value is None else str(value)
    if not raw:
        return "n/a"
    if raw in PROVENANCE_DISPLAY_LABELS:
        return PROVENANCE_DISPLAY_LABELS[raw]
    cleaned = raw.replace("_not_market_observation", "").replace("_", " ").strip()
    return cleaned.title() if cleaned else raw


def coerce_table_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce object-like numeric columns so Streamlit column formats apply."""
    if frame.empty:
        return frame
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def format_scanner_table_for_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Return scanner rows with display-safe precision for the Streamlit grid."""
    table = coerce_table_numeric_columns(frame, SCANNER_NUMERIC_COLUMNS)
    if table.empty:
        return table
    out = table.copy()
    for column in SCANNER_PERCENT_COLUMNS:
        if column in out:
            out[column] = out[column].map(lambda value: fmt_pct(value, 2))
    if "strike" in out:
        out["strike"] = out["strike"].map(fmt_money)
    for column in ("dte", "volume", "open_interest"):
        if column in out:
            out[column] = out[column].map(fmt_int)
    for column in ("residual_z_score", "liquidity_score"):
        if column in out:
            out[column] = out[column].map(lambda value: fmt_decimal(value, 2))
    return out


def fmt_decimal(value: object, digits: int = 3) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _render_metric_grid(metrics: list[tuple[str, str, str, str | None]]) -> str:
    cards = []
    for label, value, detail, help_text in metrics:
        detail_markup = f'<div class="metric-card-detail">{_html(detail)}</div>' if detail else ""
        cards.append(
            f'<div class="metric-card" title="{_html(help_text or "")}">'
            f'<div class="metric-card-label">{_html(label)}</div>'
            f'<div class="metric-card-value">{_html(value)}</div>'
            f"{detail_markup}</div>"
        )
    return f'<div class="metric-grid" data-dashboard-section="kpi-grid">{"".join(cards)}</div>'


def _quality_rows(rows: list[tuple[str, str, str | None]]) -> str:
    return "".join(
        f'<div class="quality-item"><div class="quality-item-label">{_html(label)}</div>'
        f'<div class="quality-item-value">{_html(value)}</div>'
        f'<div class="quality-item-note">{_html(note or "")}</div></div>'
        for label, value, note in rows
    )


def _bucket_count(value: object) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _render_quality_workstation(
    *,
    groups: list[tuple[str, list[tuple[str, str, str | None]]]],
    reason_buckets: dict,
    alert_level: str,
    alert_message: str,
) -> str:
    chips = "".join(
        f'<span class="quality-chip">{_html(reason)} <strong>{_html(count)}</strong></span>'
        for reason, count in sorted(reason_buckets.items(), key=lambda item: (-_bucket_count(item[1]), str(item[0])))
        if count
    )
    if not chips:
        chips = '<span class="quality-chip quality-chip-muted">no active reason buckets</span>'

    group_markup = "".join(
        f'<section class="quality-group"><div class="quality-group-title">{_html(title)}</div>'
        f'<div class="quality-items">{_quality_rows(rows)}</div></section>'
        for title, rows in groups
    )
    return f"""
    <div class="quality-workstation" data-dashboard-section="quality-summary">
        <div class="quality-alert quality-alert-{_html(alert_level)}">{_html(alert_message)}</div>
        <div class="quality-chip-row">{chips}</div>
        <div class="quality-group-grid">{group_markup}</div>
    </div>
    """


def _data_mode_class(mode: object) -> str:
    lowered = str(mode or "").lower()
    if "synthetic" in lowered:
        return "status-synthetic"
    if "fallback" in lowered or "unavailable" in lowered:
        return "status-fallback"
    return "status-live"


def _render_workstation_header(
    *,
    symbol: str,
    spot: str,
    atm_iv: str = "n/a",
    risk_reversal: str = "n/a",
    term_spread: str = "n/a",
    surface_points: str = "n/a",
    stale_age: str = "n/a",
    source: str,
    market_state: str,
    market_reason: str,
    updated: str,
    model_label: str = "BSM with dividends",
    readiness_label: str,
    readiness_detail: str,
    status_markup: str,
) -> str:
    ticker_tiles = [
        ("ATM IV", atm_iv),
        ("Risk Rev", risk_reversal),
        ("Term Spread", term_spread),
        ("Surface Points", surface_points),
        ("Stale Age", stale_age),
    ]
    tile_markup = "".join(
        f'<div class="kpi-strip-tile"><div class="kpi-strip-label">{_html(label)}</div>'
        f'<div class="kpi-strip-value">{_html(value)}</div></div>'
        for label, value in ticker_tiles
    )
    function_keys = "".join(
        f'<span class="function-key-item"><strong>F{index + 1}</strong> {_html(name)}</span>'
        for index, name in enumerate(FUNCTION_KEY_NAMES)
    )
    return f"""
    <div class="workstation-header">
        <div class="workstation-topline">
            <div class="brand-cluster">
                <div class="brand-mark">VS.</div>
                <div class="workstation-title">Options Volatility Surface Workstation</div>
                <span class="env-tag">PROD</span>
            </div>
            <div class="header-cluster">
                <span class="workstation-clock">{_html(updated)}</span>
                <span class="status-pill {_data_mode_class(market_state)}">Market: {_html(market_state)}</span>
                <span class="status-pill status-live">Model: {_html(model_label)}</span>
                <span class="shortcut-key">?</span>
            </div>
        </div>
        <div class="ticker-strip">
            <div class="workstation-symbol-block">
                <div class="workstation-symbol">{_html(symbol)}</div>
                <div class="workstation-spot">{_html(spot)}</div>
                <div class="spot-delta">{_html(surface_points)} pts</div>
            </div>
            {tile_markup}
        </div>
        <div class="function-key-strip">{function_keys}</div>
        <div class="workstation-tape">
            <span>Source <strong>{_html(source)}</strong></span>
            <span>Market <strong>{_html(market_state)}</strong></span>
            <span>Reason <strong>{_html(market_reason)}</strong></span>
            <span>Updated <strong>{_html(updated)}</strong></span>
            <span>Readiness <strong>{_html(readiness_label)}</strong></span>
        </div>
        <div class="workstation-readiness">
            <div class="readiness-title">Surface Readiness</div>
            <div class="readiness-detail">{_html(readiness_detail)}</div>
        </div>
        <div class="status-rail">{status_markup}</div>
    </div>
    """


def _mode_dot_class(mode: object) -> str:
    lowered = str(mode or "").lower()
    if "synthetic" in lowered:
        return "synthetic"
    if "fallback" in lowered or "unavailable" in lowered:
        return "fallback"
    if "stale" in lowered:
        return "stale"
    return "live"


def _render_symbol_chips(symbols: list[str], active_symbol: str, mode: object) -> str:
    chips = []
    dot_class = _mode_dot_class(mode)
    for symbol in symbols[:12]:
        active = " active" if symbol == active_symbol else ""
        chips.append(
            f'<span class="rail-chip{active}"><span class="rail-dot {dot_class}"></span>{_html(symbol)}</span>'
        )
    return f'<div class="rail-chip-row">{"".join(chips)}</div>'


def render_command_rail(
    st_module,
    *,
    available_symbols: list[str],
    watchlist_presets: dict,
    connector,
    control_help: dict,
    model_choices,
) -> dict:
    """Render the dense left command rail and return selected control state."""
    with st_module.sidebar:
        st_module.markdown('<div class="rail-heading">Symbol Command</div>', unsafe_allow_html=True)
        symbol_query = st_module.text_input(
            "Symbol command",
            value="",
            placeholder="Cmd+K Search ticker or watchlist...",
            label_visibility="collapsed",
        ).strip().upper()
        preset_name = st_module.selectbox(
            "Watchlist preset",
            options=["Custom", *watchlist_presets.keys()],
            index=1 if watchlist_presets else 0,
            help=control_help["watchlist_preset"],
        )
        preset_symbols = watchlist_presets.get(preset_name, []) if preset_name != "Custom" else []
        default_symbols = preset_symbols or ["AAPL", "MSFT", "TSLA", "NVDA", "SPY"]
        universe_options = sorted({*available_symbols, *default_symbols})
        suggestions = [symbol for symbol in universe_options if not symbol_query or symbol.startswith(symbol_query)]
        if symbol_query and suggestions:
            selected_suggestion = st_module.selectbox(
                "Ticker suggestions",
                suggestions[:8],
                index=0,
                help="Type-ahead suggestions sourced from the configured dashboard universe.",
            )
            if selected_suggestion not in default_symbols:
                default_symbols = [selected_suggestion, *default_symbols]
        selected_symbols = st_module.multiselect(
            "Universe",
            options=universe_options,
            default=[symbol for symbol in default_symbols if symbol in universe_options],
            help=control_help["universe"],
        )
        selected_symbols = [str(symbol).upper() for symbol in selected_symbols]
        if selected_symbols:
            active_symbol = str(st_module.session_state.get("active_symbol", selected_symbols[0])).upper()
            if active_symbol not in selected_symbols:
                active_symbol = selected_symbols[0]
            st_module.session_state["active_symbol"] = active_symbol
            st_module.markdown(
                _render_symbol_chips(selected_symbols, active_symbol, "live"),
                unsafe_allow_html=True,
            )
            chip_columns = st_module.columns(min(4, len(selected_symbols)))
            for index, symbol in enumerate(selected_symbols[:8]):
                with chip_columns[index % len(chip_columns)]:
                    if st_module.button(symbol, key=f"rail_symbol_{symbol}", width="stretch"):
                        st_module.session_state["active_symbol"] = symbol
                        active_symbol = symbol
        else:
            active_symbol = ""
            st_module.markdown('<div class="rail-chip-row"></div>', unsafe_allow_html=True)

        context_slot = st_module.empty()
        context_slot.markdown(
            render_command_rail_context(
                symbol=active_symbol or "N/A",
                spot="pending",
                session_state="pending",
                updated="pending",
                source="pending",
            ),
            unsafe_allow_html=True,
        )

        with st_module.expander("View - 5 active", expanded=True):
            show_3d_surface = st_module.checkbox("3D surface", value=True, help=control_help["show_3d_surface"])
            surface_x_axis = st_module.selectbox(
                "Surface x-axis",
                options=["Strike", "Moneyness", "Log-moneyness", "Call delta"],
                index=0,
                help=control_help["surface_x_axis"],
            )
            show_reliability_overlay = st_module.checkbox(
                "Reliability overlay",
                value=True,
                help="Color and size raw quote points by deterministic fit weight/reliability metadata.",
            )
            show_correlations = st_module.checkbox(
                "Realized correlation",
                value=True,
                help=control_help["show_correlations"],
            )
            show_chain = st_module.checkbox("Option chain", value=True, help=control_help["show_chain"])

        with st_module.expander("Fit - preset + guardrails", expanded=True):
            selected_fit_mode = st_module.selectbox(
                "Fit Mode",
                options=list(FIT_MODE_CHOICES),
                index=0,
                help="Select the dashboard view for fitted, prior-assisted, ML research, or diagnostic raw surface context.",
            )
            fit_preset = st_module.selectbox(
                "Fit preset",
                options=list(FIT_PRESETS),
                index=0,
                help=control_help["fit_preset"],
            )
            fit_defaults = FIT_PRESETS[fit_preset]
            fit_preset_key = fit_preset.lower().replace(" ", "_")
            fit_max_spread_pct = st_module.slider(
                "Fit max spread percent",
                0.05,
                1.50,
                float(fit_defaults["max_bid_ask_spread_pct"]),
                0.05,
                key=f"fit_max_spread_pct_{fit_preset_key}",
                help=control_help["fit_max_spread_pct"],
            )
            fit_max_quote_age_days = st_module.number_input(
                "Fit max quote age days",
                min_value=0,
                value=int(fit_defaults["max_quote_age_days"]),
                step=1,
                key=f"fit_max_quote_age_days_{fit_preset_key}",
                help=control_help["fit_max_quote_age_days"],
            )
            fit_min_volume = st_module.number_input(
                "Fit min volume",
                min_value=0,
                value=int(fit_defaults["min_volume"]),
                step=5,
                key=f"fit_min_volume_{fit_preset_key}",
                help=control_help["fit_min_volume"],
            )
            fit_min_open_interest = st_module.number_input(
                "Fit min open interest",
                min_value=0,
                value=int(fit_defaults["min_open_interest"]),
                step=10,
                key=f"fit_min_open_interest_{fit_preset_key}",
                help=control_help["fit_min_open_interest"],
            )
            fit_moneyness_band = st_module.slider(
                "Fit moneyness",
                0.35,
                2.50,
                fit_defaults["moneyness"],
                0.05,
                key=f"fit_moneyness_{fit_preset_key}",
                help=control_help["fit_moneyness"],
            )
            fit_max_raw_iv = st_module.slider(
                "Fit max raw IV",
                0.50,
                5.00,
                float(fit_defaults["max_raw_iv"]),
                0.05,
                format="%.2f",
                key=f"fit_max_raw_iv_{fit_preset_key}",
                help=control_help["fit_max_raw_iv"],
            )
            fit_no_arbitrage_policy = st_module.selectbox(
                "Fit no-arb policy",
                options=["exclude", "penalize", "allow"],
                index=["exclude", "penalize", "allow"].index(str(fit_defaults["no_arbitrage_policy"])),
                key=f"fit_no_arbitrage_policy_{fit_preset_key}",
                help=control_help["fit_no_arbitrage_policy"],
            )
            fit_last_only_policy = st_module.selectbox(
                "Fit last-only policy",
                options=["allow_penalized", "exclude", "allow"],
                index=["allow_penalized", "exclude", "allow"].index(str(fit_defaults["last_only_policy"])),
                key=f"fit_last_only_policy_{fit_preset_key}",
                help=control_help["fit_last_only_policy"],
            )

        with st_module.expander("Chain filters - 5 active", expanded=False):
            max_spread_pct = st_module.slider(
                "Max spread percent",
                0.05,
                1.50,
                0.75,
                0.05,
                help=control_help["max_spread_pct"],
            )
            min_open_interest = st_module.number_input(
                "Min open interest",
                min_value=0,
                value=0,
                step=10,
                help=control_help["min_open_interest"],
            )
            min_volume = st_module.number_input(
                "Min volume",
                min_value=0,
                value=0,
                step=5,
                help=control_help["min_volume"],
            )
            max_quote_age_days = st_module.number_input(
                "Max quote age days",
                min_value=0,
                value=5,
                step=1,
                help=control_help["max_quote_age_days"],
            )
            option_price_source = st_module.selectbox(
                "IV price source",
                options=["mark", "midpoint", "last"],
                index=0,
                help=control_help["option_price_source"],
            )
            pricing_model = st_module.selectbox(
                "Pricing model",
                options=list(model_choices),
                index=1,
                help=control_help["pricing_model"],
            )

        with st_module.expander("Refresh - scheduler", expanded=True):
            auto_refresh = st_module.checkbox("Auto refresh", value=False, help=control_help["auto_refresh"])
            refresh_interval = st_module.slider(
                "Refresh interval seconds",
                15,
                180,
                60,
                15,
                help=control_help["refresh_interval"],
            )
            if st_module.button("Refresh data", width="stretch"):
                result = connector.trigger_data_refresh()
                st_module.cache_data.clear()
                if result.get("status") == "success":
                    st_module.success(result.get("message", "Data refreshed"))
                else:
                    st_module.error(result.get("message", "Refresh failed"))

        st_module.markdown(
            """
            <div class="rail-footer">
                <strong>1-0</strong> tabs &nbsp; <strong>R</strong> refresh &nbsp;
                <strong>/</strong> symbol &nbsp; <strong>?</strong> help<br>
                v1.x.y build · <a href="https://github.com/" target="_blank">docs</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "context_slot": context_slot,
        "preset_name": preset_name,
        "selected_symbols": selected_symbols,
        "surface_symbol": active_symbol,
        "show_3d_surface": show_3d_surface,
        "surface_x_axis": surface_x_axis,
        "selected_fit_mode": selected_fit_mode,
        "show_reliability_overlay": show_reliability_overlay,
        "show_correlations": show_correlations,
        "show_chain": show_chain,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
        "max_spread_pct": max_spread_pct,
        "min_open_interest": min_open_interest,
        "min_volume": min_volume,
        "max_quote_age_days": max_quote_age_days,
        "option_price_source": option_price_source,
        "pricing_model": pricing_model,
        "fit_preset": fit_preset,
        "fit_max_spread_pct": fit_max_spread_pct,
        "fit_max_quote_age_days": fit_max_quote_age_days,
        "fit_min_volume": fit_min_volume,
        "fit_min_open_interest": fit_min_open_interest,
        "fit_moneyness_band": fit_moneyness_band,
        "fit_max_raw_iv": fit_max_raw_iv,
        "fit_no_arbitrage_policy": fit_no_arbitrage_policy,
        "fit_last_only_policy": fit_last_only_policy,
    }


def render_command_rail_context(
    *,
    symbol: str,
    spot: str,
    session_state: str,
    updated: str,
    source: str,
) -> str:
    return f"""
    <div class="rail-panel rail-context">
        <div class="rail-command-label">Active Context</div>
        <div class="rail-context-symbol">{_html(symbol)}</div>
        <div class="rail-context-spot">{_html(spot)}</div>
        <div class="rail-context-meta">Session {_html(session_state)} · Refresh {_html(updated)}</div>
        <div class="rail-context-meta">Source {_html(source)}</div>
    </div>
    """


FUNCTION_KEY_NAMES = (
    "Surface",
    "Chain",
    "Skew",
    "Term",
    "Quality",
    "Scanner",
    "Strategy",
    "Risk",
    "Diag",
    "Export",
)


def function_key_page_titles(page_registry: list) -> list[str]:
    """Return Bloomberg-style function-key tab labels without changing page specs."""
    return [
        f"F{index + 1} {FUNCTION_KEY_NAMES[index] if index < len(FUNCTION_KEY_NAMES) else page.title}"
        for index, page in enumerate(page_registry)
    ]


def fit_mode_state(selected_mode: str, surface_meta: dict) -> dict:
    """Return dashboard fit-mode state sourced from existing metadata."""
    selected = selected_mode if selected_mode in FIT_MODE_CHOICES else "Robust"
    prior_applied = bool(surface_meta.get("surface_prior_applied"))
    mode_rows = surface_meta.get("fit_mode_comparison") or []
    mode_by_name = {str(row.get("mode")): row for row in mode_rows}
    mapping = {
        "Robust": ("Robust SVI", CURRENT_ROBUST_FIT_PROVENANCE),
        "Standard": ("Standard SVI", STANDARD_SVI_FIT_PROVENANCE),
        "Prior Assisted": ("Prior Assisted", PRIOR_ASSISTED_FIT_PROVENANCE),
        "ML Denoised": ("ML Denoised", ML_DENOISED_PROVENANCE),
        "Diagnostic Raw": ("Diagnostic Raw", RAW_QUOTE_DIAGNOSTIC_PROVENANCE),
    }
    row_name, estimate_type = mapping[selected]
    row = mode_by_name.get(row_name, {})
    available = True
    if selected == "Prior Assisted":
        available = prior_applied or bool(row)
    elif selected == "ML Denoised":
        available = bool(row.get("enabled")) if row else False
    elif selected == "Diagnostic Raw":
        available = bool(surface_meta.get("svi_smiles") or surface_meta.get("fit_diagnostics"))
    warning = ""
    if selected == "ML Denoised" and not available:
        warning = "ML Denoised is research-only and off by default; robust deterministic estimates remain displayed."
    elif selected == "Prior Assisted" and not prior_applied:
        warning = "Prior Assisted was not applied; current robust fit estimates remain displayed."
    elif selected == "Standard":
        warning = "Standard mode is shown for comparison; reliability-weighted robust estimates remain the active surface."
    elif selected == "Diagnostic Raw":
        warning = "Diagnostic Raw emphasizes observed quote points and reliability; fitted surface estimates remain visible for context."
    return {
        "selected_mode": selected,
        "comparison_row": row,
        "estimate_type": estimate_type,
        "available": bool(available),
        "chart_label": f"{selected} View",
        "provenance": row.get("provenance") or surface_meta.get("surface_estimate_type") or estimate_type,
        "warning": warning,
    }


def fit_comparison_display_rows(surface_meta: dict) -> list[dict]:
    """Normalize fit-mode metadata into the Phase 7 comparison table shape."""
    rows = []
    timestamp = surface_meta.get("timestamp") or surface_meta.get("spot_timestamp")
    if hasattr(timestamp, "isoformat"):
        timestamp = timestamp.isoformat()
    fit_eligible = surface_meta.get("fit_eligible_count")
    fit_excluded = surface_meta.get("fit_excluded_count")
    for row in surface_meta.get("fit_mode_comparison") or []:
        rows.append(
            {
                "fit_mode": row.get("mode"),
                "status": row.get("status"),
                "eligible_rows": row.get("eligible_rows", fit_eligible),
                "excluded_rows": row.get("excluded_rows", fit_excluded),
                "weighted_rmse": row.get("weighted_rmse"),
                "unweighted_rmse": row.get("unweighted_rmse", row.get("rmse")),
                "no_arb_violations": row.get("arbitrage_violations", surface_meta.get("no_arbitrage_violation_count")),
                "prior_weight": row.get("prior_weight", surface_meta.get("surface_prior_blend_weight")),
                "ml_uncertainty": row.get("uncertainty"),
                "timestamp": timestamp,
                "provenance": display_provenance_label(row.get("provenance") or surface_meta.get("surface_source")),
            }
        )
    return rows


def data_quality_actionability(surface_meta: dict) -> dict:
    """Summarize actionable quality diagnostics from metadata only."""
    reason_buckets = dict(surface_meta.get("quality_reason_buckets") or surface_meta.get("rejection_reasons") or {})
    penalty_buckets = dict(surface_meta.get("fit_penalty_reason_buckets") or {})
    hard_buckets = dict(surface_meta.get("fit_hard_rejection_reason_buckets") or {})
    combined = {}
    for source in (reason_buckets, penalty_buckets, hard_buckets):
        for reason, count in source.items():
            if count:
                combined[str(reason)] = combined.get(str(reason), 0) + int(count)
    top_penalties = [
        {"reason": reason, "count": count}
        for reason, count in sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]
    expiry_rows = []
    for expiry, payload in sorted((surface_meta.get("expiry_quality") or {}).items()):
        expiry_rows.append(
            {
                "expiry": expiry,
                "score": payload.get("score"),
                "surface_quotes": payload.get("surface_quotes"),
                "rejected_quotes": payload.get("rejected_quotes"),
                "reason_buckets": payload.get("reason_buckets") or {},
            }
        )
    expiry_rows.sort(key=lambda row: (999.0 if row["score"] is None else float(row["score"]), str(row["expiry"])))
    residuals = ((surface_meta.get("fit_diagnostics") or {}).get("residual_diagnostics") or {}).get("top_residuals") or []
    no_arb = {
        "violation_count": surface_meta.get("no_arbitrage_violation_count"),
        "violation_rows": surface_meta.get("no_arbitrage_violation_rows"),
        "excluded_count": surface_meta.get("no_arbitrage_excluded_count"),
        "reason_buckets": surface_meta.get("no_arbitrage_reason_buckets") or {},
        "post_fit": (surface_meta.get("post_fit_arbitrage") or {}).get("reason_buckets") or {},
    }
    suggest_strict = bool(
        (surface_meta.get("surface_quality_score") is not None and float(surface_meta.get("surface_quality_score")) < 80.0)
        or no_arb.get("violation_rows")
        or hard_buckets
    )
    return {
        "top_penalty_reasons": top_penalties,
        "worst_expiries": expiry_rows[:5],
        "worst_residual_contracts": residuals[:8],
        "no_arbitrage": no_arb,
        "suggested_preset": "Strict" if suggest_strict else surface_meta.get("fit_filter_preset") or "Standard",
    }


def quality_drop_alert_summary(surface_meta: dict) -> dict:
    """Return alert copy for material quality changes versus persisted snapshots."""
    alert = surface_meta.get("quality_drop_alert") or {}
    if not alert.get("available"):
        return {"level": "info", "message": alert.get("reason") or "No prior quality snapshot is available."}
    score_change = alert.get("score_change")
    deltas = alert.get("reason_bucket_delta") or {}
    if alert.get("triggered"):
        drivers = ", ".join(f"{key}: +{value}" for key, value in deltas.items() if value > 0) or "score drop"
        return {
            "level": "warning",
            "message": f"Quality dropped {score_change:.1f} points versus {alert.get('previous_snapshot_timestamp')}; likely drivers: {drivers}.",
        }
    return {
        "level": "success",
        "message": f"Quality is stable versus {alert.get('previous_snapshot_timestamp')}; score change {score_change:.1f}.",
    }


def fit_diagnostics_export_payload(symbol: str, surface_meta: dict) -> dict:
    """Build a reproducible diagnostics export payload without recomputing metrics."""
    return {
        "symbol": symbol,
        "selected_surface_mode": surface_meta.get("selected_fit_mode"),
        "fit_mode_comparison": surface_meta.get("fit_mode_comparison") or [],
        "fit_diagnostics": surface_meta.get("fit_diagnostics") or {},
        "global_fit_diagnostics": surface_meta.get("global_fit_diagnostics") or {},
        "surface_quality": surface_meta.get("surface_quality") or {},
        "post_fit_arbitrage": surface_meta.get("post_fit_arbitrage") or {},
        "surface_prior": surface_meta.get("surface_prior") or {},
        "surface_repair": surface_meta.get("surface_repair") or {},
        "row_weights": _fit_residual_rows_for_export(surface_meta),
        "provenance": {
            "surface_source": surface_meta.get("surface_source"),
            "surface_mode": surface_meta.get("surface_mode"),
            "surface_estimate_type": surface_meta.get("surface_estimate_type"),
            "option_price_source": surface_meta.get("option_price_source"),
            "pricing_model": surface_meta.get("pricing_model_label"),
        },
    }


def _fit_residual_rows_for_export(surface_meta: dict) -> list[dict]:
    rows = []
    for smile in surface_meta.get("svi_smiles") or []:
        for row in smile.get("residuals") or []:
            rows.append({**row, "expiration": smile.get("expiration"), "dte": smile.get("dte")})
    return rows


def run_dashboard() -> None:

    import json
    import os
    import sys
    import time
    import warnings
    from datetime import datetime
    from typing import Any, Dict, Iterable, Tuple

    import numpy as np
    import plotly.graph_objects as go
    import streamlit as st
    from src.dashboard.components import card
    from src.dashboard.formatting import fmt_int, fmt_money, fmt_pct
    from src.dashboard.keyboard import render_keyboard_layer
    from src.dashboard.loading import LoadingState, load_with_status, render_empty_state
    from src.dashboard.pages import default_page_registry
    from src.dashboard.state import DashboardStateService
    from src.dashboard.surface_view import extract_smile, surface_axis, surface_stats
    from src.dashboard.tables import add_freshness_column, dataframe_to_csv_bytes, filter_market_snapshot, filter_option_chain
    from src.dashboard.theme import apply_chart_layout, inject_theme, status_pill
    from src.dashboard.tooltips import COLUMN_HELP, CONTROL_HELP, KPI_HELP
    from src.quant.model_selection import MODEL_CHOICES

    warnings.filterwarnings("ignore")

    st.set_page_config(
        page_title="Vol Surface Workstation",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    inject_theme(st)


    CONNECTOR_AVAILABLE = False
    REAL_SYSTEM_AVAILABLE = False

    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if project_root not in sys.path:
            sys.path.append(project_root)
        from dashboard_connector import DashboardConnector

        CONNECTOR_AVAILABLE = True
        REAL_SYSTEM_AVAILABLE = True
    except ImportError as exc:
        st.sidebar.warning(f"Dashboard connector unavailable: {exc}")


    class MinimalFallbackConnector:
        """Deterministic local fallback used only if the real connector cannot load."""

        def __init__(self):
            self.timestamp = datetime.now()
            self.app_test_mode = os.environ.get("VOL_SURFACE_APPTEST_MODE", "fallback").strip().lower()

        def _mode(self) -> str:
            if self.app_test_mode == "synthetic":
                return "Synthetic"
            return "Fallback"

        def _source(self) -> str:
            if self.app_test_mode == "synthetic":
                return "deterministic demo provider"
            if self.app_test_mode == "provider_failure":
                return "forced provider failure"
            return "static fallback"

        def _fallback_reason(self) -> str:
            if self.app_test_mode == "provider_failure":
                return "Forced provider failure for deterministic AppTest coverage"
            if self.app_test_mode == "synthetic":
                return "Real option chain unavailable in deterministic AppTest mode"
            return "DashboardConnector could not be imported"

        def get_current_data(self, symbol: str) -> Dict[str, Any]:
            prices = {"AAPL": 196.50, "MSFT": 416.50, "NVDA": 138.50, "TSLA": 325.00, "SPY": 578.00}
            vols = {"AAPL": 0.25, "MSFT": 0.22, "NVDA": 0.40, "TSLA": 0.50, "SPY": 0.15}
            spot = prices.get(symbol.upper(), 100.0)
            iv = vols.get(symbol.upper(), 0.30)
            return {
                "symbol": symbol.upper(),
                "price": spot,
                "price_source": self._source(),
                "data_mode": self._mode(),
                "iv_source": self._source(),
                "volume": None,
                "open_interest": None,
                "iv_30d": iv,
                "iv_60d": iv * 1.05,
                "iv_90d": iv * 1.10,
                "delta": 0.52,
                "gamma": 0.018,
                "theta": -0.08,
                "vega": 0.22,
                "dividend_yield_30d": 0.0,
                "corporate_action_warning_count": 0,
                "bid_ask_spread": None,
                "contracts": None,
                "market_status": self.get_market_status().get("session_state"),
                "market_reason": self.get_market_status().get("reason"),
                "data_delay_minutes": self.get_market_status().get("data_delay_minutes"),
                "timestamp": self.timestamp,
            }

        def get_vol_surface_data(self, symbol: str):
            spot = self.get_current_data(symbol)["price"]
            strikes = np.linspace(spot * 0.8, spot * 1.2, 16)
            expiries = np.array([7, 14, 30, 60, 90, 180])
            surface = np.zeros((len(expiries), len(strikes)))
            base = self.get_current_data(symbol)["iv_30d"]
            for i, days in enumerate(expiries):
                for j, strike in enumerate(strikes):
                    m = strike / spot
                    surface[i, j] = base * (1 + 0.08 * np.sqrt(days / 365)) - 0.08 * (m - 1) + 0.04 * (m - 1) ** 2
            return strikes, expiries, surface

        def get_surface_metadata(self, symbol: str):
            return {
                "symbol": symbol.upper(),
                "mode": self._mode(),
                "surface_mode": self._mode(),
                "surface_source": self._source(),
                "timestamp": self.timestamp,
                "raw_rows": 0,
                "valid_rows": 0,
                "rejected_rows": 0,
                "data_quality_score": 0.0,
                "surface_quality_score": 0.0,
                "quality_reason_buckets": {},
                "expiry_quality": {},
                "surface_quality": {
                    "score": 0.0,
                    "valid_quotes": 0,
                    "rejected_quotes": 0,
                    "surface_quotes": 96,
                    "no_arbitrage_excluded_quotes": 0,
                    "reason_buckets": {},
                    "expiries": {},
                },
                "no_arbitrage_violation_count": 0,
                "no_arbitrage_violation_rows": 0,
                "no_arbitrage_excluded_count": 0,
                "no_arbitrage_reason_buckets": {},
                "fallback_reason": self._fallback_reason(),
                "surface_points": 96,
                "pricing_model": "bsm_dividends",
                "pricing_model_label": "BSM with dividends",
                "pricing_model_assumptions": "Static fallback model metadata.",
                "contract_greeks_count": 0,
                "greek_units": {},
            }

        def get_options_chain_snapshot(self, symbol: str):
            return pd.DataFrame(), self.get_surface_metadata(symbol)

        def get_market_data_snapshot(self, symbol: str):
            from src.data.models import MarketDataSnapshot

            data = self.get_current_data(symbol)
            return MarketDataSnapshot(
                symbol=symbol.upper(),
                spot=float(data["price"]),
                spot_timestamp=self.timestamp,
                chain_timestamp=self.timestamp,
                source=self._source(),
                fallback_reason=self._fallback_reason(),
                mode=self._mode(),
            )

        def get_portfolio_metrics(self, position_csv=None):
            return {"configured": False, "message": "No position book configured"}

        def get_portfolio_optimization(self, position_csv, objective: str = "delta-neutral", theta_target: float = 0.0):
            return {"available": False, "reason": "No option-chain portfolio provider in fallback mode"}

        def get_correlation_matrix(self, symbols: Iterable[str], period: str = "6mo"):
            return pd.DataFrame()

        def get_historical_metrics(self, symbol: str, period: str = "1y"):
            return {"available": False, "reason": "No historical provider in fallback mode"}

        def get_relative_value_dashboard(self, left_symbol: str, right_symbol: str):
            from src.quant.advanced_features import relative_value_dashboard

            return relative_value_dashboard(self.get_current_data(left_symbol), self.get_current_data(right_symbol))

        def get_cross_sectional_vol_map(self, symbols: Iterable[str]):
            from src.quant.advanced_features import cross_sectional_vol_map

            return cross_sectional_vol_map([self.get_current_data(symbol) for symbol in symbols])

        def get_earnings_event_engine(self, symbol: str):
            return {"available": False, "reason": "No event provider in fallback mode"}

        def get_strategy_analytics(self, symbol: str, strategy_type: str):
            return {"available": False, "reason": "No option-chain strategy provider in fallback mode"}

        def get_strategy_scenarios(self, symbol: str, strategy_type: str, **kwargs):
            return {"available": False, "reason": "No option-chain strategy scenario provider in fallback mode"}

        def get_surface_alerts(self, symbol: str, **kwargs):
            return {"available": True, "configured": {}, "alert_count": 0, "alerts": []}

        def get_watchlist_presets(self):
            from src.quant.advanced_features import watchlist_presets

            return watchlist_presets()

        def generate_research_report(self, symbol: str, path=None, **kwargs):
            from pathlib import Path

            from src.quant.advanced_features import generate_research_report

            output = Path(path) if path is not None else Path("reports") / f"{symbol.upper()}_surface_report.html"
            return generate_research_report(
                {
                    "symbol": symbol.upper(),
                    "data_timestamp": self.timestamp.isoformat(),
                    "model_assumptions": "Static fallback model metadata.",
                    "surface_summary": self.get_current_data(symbol),
                    "diagnostics": self.get_surface_metadata(symbol),
                },
                output,
                timestamp=self.timestamp,
            )

        def get_ml_anomaly_detector(self, symbol: str, observations=None):
            from src.quant.advanced_features import ml_anomaly_detector

            return ml_anomaly_detector(observations or [])

        def get_vol_regime_classifier(self, symbol: str, observations=None):
            from src.quant.advanced_features import classify_vol_regime

            return classify_vol_regime(observations or [], current=self.get_current_data(symbol))

        def get_forecasting_module(self, symbol: str, observations=None):
            from src.quant.advanced_features import forecast_volatility

            return forecast_volatility(observations or [])

        def get_news_event_overlay(self, symbol: str, surface_jumps=None):
            from src.quant.advanced_features import news_event_overlay

            return news_event_overlay([], surface_jumps or [])

        def request_async_refresh(self, symbol: str):
            self.timestamp = datetime.now()
            return {"key": symbol.upper(), "status": "scheduled", "already_running": False}

        def get_async_refresh_status(self, symbol: str | None = None):
            key = symbol.upper() if symbol else "fallback"
            return {"available": True, "source": "fallback_refresh", "refreshes": {key: {"status": "complete"}}}

        def get_market_status(self):
            from src.data.market_calendar import MarketCalendar

            return MarketCalendar().status(self.timestamp).as_dict()

        def configure_liquidity_filters(self, **kwargs):
            return {}

        def configure_option_price_source(self, price_source: str):
            return price_source

        def configure_pricing_model(self, pricing_model: str):
            return pricing_model

        def configure_fit_filters(self, **kwargs):
            return dict(kwargs)

        def get_system_health(self):
            return {
                "overall": {
                    "yfinance_available": False,
                    "last_update": self.timestamp,
                    "option_chain_cache_entries": 0,
                    "market_state": self.get_market_status().get("session_state"),
                },
                "performance": {"real_time_active": False, "update_interval": 30},
                "data_contract": {"price_provider": "static fallback", "calendar_provider": "XNYS fallback"},
                "app_test_mode": self.app_test_mode,
            }

        def trigger_data_refresh(self):
            self.timestamp = datetime.now()
            return {"status": "success", "message": "Fallback timestamp refreshed"}

        def start_real_time_updates(self):
            return False


    @st.cache_resource
    def init_dashboard_system():
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return MinimalFallbackConnector(), False
        if CONNECTOR_AVAILABLE:
            connector = DashboardConnector()
            if REAL_SYSTEM_AVAILABLE:
                connector.start_real_time_updates()
            return connector, True
        return MinimalFallbackConnector(), False


    connector, system_ready = init_dashboard_system()


    @st.cache_data(ttl=120, show_spinner=False)
    def get_current_data_cached(symbol: str, data_key: Tuple[Any, ...]):
        return connector.get_current_data(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_vol_surface_data_cached(symbol: str, data_key: Tuple[Any, ...]):
        return connector.get_vol_surface_data(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_surface_metadata_cached(symbol: str, data_key: Tuple[Any, ...]):
        return connector.get_surface_metadata(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_options_chain_cached(symbol: str, data_key: Tuple[Any, ...]):
        return connector.get_options_chain_snapshot(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_market_snapshot_cached(symbol: str, data_key: Tuple[Any, ...]):
        return connector.get_market_data_snapshot(symbol)


    @st.cache_data(ttl=900, show_spinner=False)
    def get_correlation_matrix_cached(symbols_key: Tuple[str, ...]):
        return connector.get_correlation_matrix(list(symbols_key))


    @st.cache_data(ttl=900, show_spinner=False)
    def get_historical_metrics_cached(symbol: str):
        return connector.get_historical_metrics(symbol)


    @st.cache_data(ttl=60, show_spinner=False)
    def get_market_status_cached():
        return connector.get_market_status()

    @st.cache_data(ttl=300, show_spinner=False)
    def get_relative_value_cached(left_symbol: str, right_symbol: str, data_key: Tuple[Any, ...]):
        return connector.get_relative_value_dashboard(left_symbol, right_symbol)

    @st.cache_data(ttl=300, show_spinner=False)
    def get_cross_sectional_vol_map_cached(symbols_key: Tuple[str, ...], data_key: Tuple[Any, ...]):
        return connector.get_cross_sectional_vol_map(list(symbols_key))

    @st.cache_data(ttl=300, show_spinner=False)
    def get_earnings_event_cached(symbol: str, data_key: Tuple[Any, ...]):
        return connector.get_earnings_event_engine(symbol)

    @st.cache_data(ttl=300, show_spinner=False)
    def get_strategy_analytics_cached(symbol: str, strategy_type: str, data_key: Tuple[Any, ...]):
        return connector.get_strategy_analytics(symbol, strategy_type)

    @st.cache_data(ttl=300, show_spinner=False)
    def get_strategy_scenarios_cached(
        symbol: str,
        strategy_type: str,
        spot_axis_key: Tuple[float, ...],
        time_axis_key: Tuple[float, ...],
        vol_axis_key: Tuple[float, ...],
        skew_axis_key: Tuple[float, ...],
        data_key: Tuple[Any, ...],
    ):
        return connector.get_strategy_scenarios(
            symbol,
            strategy_type,
            spot_shifts=list(spot_axis_key),
            time_pass_days=list(time_axis_key),
            vol_shifts=list(vol_axis_key),
            skew_shifts=list(skew_axis_key),
        )

    @st.cache_data(ttl=300, show_spinner=False)
    def get_portfolio_metrics_cached(position_csv: bytes | None, data_key: Tuple[Any, ...]):
        return connector.get_portfolio_metrics(position_csv)

    @st.cache_data(ttl=300, show_spinner=False)
    def get_portfolio_optimization_cached(
        position_csv: bytes | None,
        objective: str,
        theta_target: float,
        data_key: Tuple[Any, ...],
    ):
        return connector.get_portfolio_optimization(position_csv, objective, theta_target=theta_target)

    @st.cache_data(ttl=120, show_spinner=False)
    def get_surface_alerts_cached(
        symbol: str,
        config_key: Tuple[float, float, float, float, float],
        data_key: Tuple[Any, ...],
    ):
        config = {
            "iv_rank_threshold": config_key[0],
            "skew_steepening_threshold": config_key[1],
            "surface_fit_error_threshold": config_key[2],
            "data_stale_minutes": config_key[3],
            "rich_cheap_residual_threshold": config_key[4],
        }
        return connector.get_surface_alerts(symbol, config=config)

    @st.cache_data(ttl=900, show_spinner=False)
    def get_watchlist_presets_cached():
        return connector.get_watchlist_presets()


    available_symbols = [
        "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA",
        "AMD", "NFLX", "CRM", "ORCL", "ADBE", "PLTR",
        "SPY", "QQQ", "IWM", "VTI",
        "JPM", "BAC", "WFC", "GS",
        "JNJ", "PFE", "UNH", "MRNA",
        "KO", "PEP", "WMT", "HD",
        "DIS", "SPOT", "COIN", "SQ", "PYPL",
        "GME", "AMC", "RBLX", "UBER", "LYFT", "F", "GM",
        "XOM", "CVX", "COP", "V", "MA", "INTC", "IBM",
        "CSCO", "BABA", "NIO", "RIVN", "LCID", "SOFI", "HOOD", "DKNG",
    ]
    watchlist_presets = get_watchlist_presets_cached()

    rail_state = render_command_rail(
        st,
        available_symbols=available_symbols,
        watchlist_presets=watchlist_presets,
        connector=connector,
        control_help=CONTROL_HELP,
        model_choices=MODEL_CHOICES,
    )
    selected_symbols = rail_state["selected_symbols"]
    surface_symbol = rail_state["surface_symbol"]
    show_3d_surface = rail_state["show_3d_surface"]
    surface_x_axis = rail_state["surface_x_axis"]
    selected_fit_mode = rail_state["selected_fit_mode"]
    show_reliability_overlay = rail_state["show_reliability_overlay"]
    show_correlations = rail_state["show_correlations"]
    show_chain = rail_state["show_chain"]
    auto_refresh = rail_state["auto_refresh"]
    refresh_interval = rail_state["refresh_interval"]
    max_spread_pct = rail_state["max_spread_pct"]
    min_open_interest = rail_state["min_open_interest"]
    min_volume = rail_state["min_volume"]
    max_quote_age_days = rail_state["max_quote_age_days"]
    option_price_source = rail_state["option_price_source"]
    pricing_model = rail_state["pricing_model"]
    fit_preset = rail_state["fit_preset"]
    fit_max_spread_pct = rail_state["fit_max_spread_pct"]
    fit_max_quote_age_days = rail_state["fit_max_quote_age_days"]
    fit_min_volume = rail_state["fit_min_volume"]
    fit_min_open_interest = rail_state["fit_min_open_interest"]
    fit_moneyness_band = rail_state["fit_moneyness_band"]
    fit_max_raw_iv = rail_state["fit_max_raw_iv"]
    fit_no_arbitrage_policy = rail_state["fit_no_arbitrage_policy"]
    fit_last_only_policy = rail_state["fit_last_only_policy"]

    if not selected_symbols:
        st.warning("Select at least one symbol from the command rail to begin analysis.")
        st.stop()
    data_key = (
        int(min_open_interest),
        int(min_volume),
        float(max_spread_pct),
        int(max_quote_age_days),
        str(option_price_source),
        str(pricing_model),
        str(fit_preset),
        float(fit_max_spread_pct),
        int(fit_max_quote_age_days),
        int(fit_min_volume),
        int(fit_min_open_interest),
        float(fit_moneyness_band[0]),
        float(fit_moneyness_band[1]),
        float(fit_max_raw_iv),
        str(fit_no_arbitrage_policy),
        str(fit_last_only_policy),
    )
    connector.configure_liquidity_filters(
        min_open_interest=data_key[0],
        min_volume=data_key[1],
        max_bid_ask_spread_pct=data_key[2],
        max_quote_age_days=data_key[3],
    )
    connector.configure_option_price_source(data_key[4])
    connector.configure_pricing_model(data_key[5])
    connector.configure_fit_filters(
        preset=data_key[6],
        max_bid_ask_spread_pct=data_key[7],
        max_quote_age_days=data_key[8],
        min_volume=data_key[9],
        min_open_interest=data_key[10],
        moneyness_min=data_key[11],
        moneyness_max=data_key[12],
        max_raw_iv=data_key[13],
        no_arbitrage_policy=data_key[14],
        last_only_policy=data_key[15],
    )
    dashboard_state = DashboardStateService()
    dashboard_state.set_context(
        selected_symbol=surface_symbol,
        selected_symbols=selected_symbols,
        data_key=data_key,
    )

    current_data = load_with_status(
        st,
        LoadingState(
            title=f"{surface_symbol} price snapshot",
            detail="Fetching yfinance underlying price, data mode, and representative IV fields.",
            stage="underlying",
            rows=4,
        ),
        lambda: get_current_data_cached(surface_symbol, data_key),
    )
    strikes, expiries, vol_surface = load_with_status(
        st,
        LoadingState(
            title=f"{surface_symbol} volatility surface",
            detail="Loading option-chain inputs and fitting the annualized IV surface.",
            stage="surface",
            rows=6,
        ),
        lambda: get_vol_surface_data_cached(surface_symbol, data_key),
    )
    surface_meta = get_surface_metadata_cached(surface_symbol, data_key)
    fit_mode_view = fit_mode_state(selected_fit_mode, surface_meta)
    surface_meta = {**surface_meta, "selected_fit_mode": fit_mode_view["selected_mode"]}
    stats = surface_stats(strikes, expiries, vol_surface, current_data["price"])
    term_metrics = stats.get("term_metrics") or {}
    market_status = get_market_status_cached()
    updated_clock = datetime.now().strftime("%H:%M:%S")

    surface_mode = surface_meta.get("surface_mode") or current_data.get("data_mode", "Unknown")
    source_label = surface_meta.get("surface_source") or current_data.get("price_source", "Unknown")
    quality_score = surface_meta.get("surface_quality_score") or surface_meta.get("data_quality_score")
    reason_buckets = surface_meta.get("quality_reason_buckets") or surface_meta.get("rejection_reasons") or {}
    quality_alert = quality_drop_alert_summary(surface_meta)
    warning_text = " | ".join(str(item) for item in (surface_meta.get("warnings") or [])[:3])
    readiness_points = stats.get("points") or surface_meta.get("surface_points")
    readiness_label = "Unavailable"
    if readiness_points:
        readiness_label = "Ready"
        if "fallback" in str(surface_mode).lower():
            readiness_label = "Fallback ready"
        elif "synthetic" in str(surface_mode).lower():
            readiness_label = "Synthetic ready"
    readiness_detail = (
        f"{fmt_int(readiness_points)} fitted grid points; "
        f"{fmt_int(surface_meta.get('fit_eligible_count'))} rows included, "
        f"{fmt_int(surface_meta.get('fit_excluded_count'))} excluded; "
        f"estimate label {display_provenance_label(fit_mode_view['provenance'])}."
    )
    if fit_mode_view.get("warning"):
        readiness_detail = f"{readiness_detail} {fit_mode_view['warning']}"

    rail_state["context_slot"].markdown(
        render_command_rail_context(
            symbol=surface_symbol,
            spot=fmt_money(current_data.get("price")),
            session_state=market_status.get("session_state", "Unknown"),
            updated=updated_clock,
            source=source_label,
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        _render_workstation_header(
            symbol=surface_symbol,
            spot=fmt_money(current_data.get("price")),
            atm_iv=fmt_pct(stats.get("atm_iv")),
            risk_reversal=fmt_pct(surface_meta.get("front_risk_reversal_25d")),
            term_spread=fmt_pct(stats.get("term_spread")),
            surface_points=fmt_int(stats.get("points")),
            stale_age=f"{fmt_int(surface_meta.get('stale_quote_count'))} stale",
            source=source_label,
            market_state=market_status.get("session_state", "Unknown"),
            market_reason=market_status.get("reason", "unknown"),
            updated=updated_clock,
            model_label=surface_meta.get("pricing_model_label") or "BSM with dividends",
            readiness_label=readiness_label,
            readiness_detail=readiness_detail,
            status_markup=(
                status_pill("Surface", surface_mode)
                + status_pill("Price", current_data.get("data_mode", "Unknown"))
                + status_pill("IV", current_data.get("iv_source", "Unknown"))
                + status_pill("Model", surface_meta.get("pricing_model_label") or "BSM with dividends")
                + status_pill("Fit View", fit_mode_view["selected_mode"])
                + status_pill("Market", market_status.get("session_state", "Unknown"))
            ),
        ),
        unsafe_allow_html=True,
    )

    kpis = [
        ("Spot", fmt_money(current_data.get("price")), current_data.get("price_source", ""), KPI_HELP.get("Spot")),
        ("ATM IV", fmt_pct(stats.get("atm_iv")), "nearest strike", KPI_HELP.get("ATM IV")),
        ("ATM dIV", fmt_pct(surface_meta.get("atm_iv_change")), "vs previous snapshot", KPI_HELP.get("ATM dIV")),
        (
            "Exp Move",
            fmt_money(surface_meta.get("front_expected_move")),
            fmt_pct(surface_meta.get("front_expected_move_pct")),
            KPI_HELP.get("Exp Move"),
        ),
        ("IV Rank", fmt_pct(surface_meta.get("iv_rank")), "stored snapshots", KPI_HELP.get("IV Rank")),
        (
            "IV Percentile",
            fmt_pct(surface_meta.get("iv_percentile")),
            "stored snapshots",
            KPI_HELP.get("IV Percentile"),
        ),
        ("Term Spread", fmt_pct(stats.get("term_spread")), "back minus front", KPI_HELP.get("Term Spread")),
        ("Surface Points", fmt_int(stats.get("points")), surface_mode, KPI_HELP.get("Surface Points")),
        (
            "Contracts",
            fmt_int(surface_meta.get("valid_rows") or current_data.get("contracts")),
            "valid rows",
            KPI_HELP.get("Contracts"),
        ),
        (
            "Quality Score",
            f"{quality_score:.1f}/100" if quality_score is not None else "n/a",
            "surface",
            KPI_HELP.get("Quality Score"),
        ),
    ]
    st.markdown(_render_metric_grid(kpis), unsafe_allow_html=True)

    st.markdown(
        _render_quality_workstation(
            groups=[
                (
                    "Chain Quality",
                    [
                        ("Score", f"{quality_score:.1f}/100" if quality_score is not None else "n/a", "surface quality"),
                        ("Raw Rows", fmt_int(surface_meta.get("raw_rows")), "provider-normalized"),
                        ("Valid Rows", fmt_int(surface_meta.get("valid_rows")), "display eligible"),
                        ("Rejected Rows", fmt_int(surface_meta.get("rejected_rows")), "normalization"),
                        ("Liquidity Rejects", fmt_int(surface_meta.get("liquidity_filtered_count")), "chain filters"),
                    ],
                ),
                (
                    "Provenance",
                    [
                        ("Surface Source", surface_meta.get("surface_source") or source_label, "provider path"),
                        ("Surface Mode", surface_mode, "live/delayed/synthetic/fallback"),
                        (
                            "Estimate Label",
                            display_provenance_label(fit_mode_view["provenance"]),
                            "not a market observation",
                        ),
                        ("Fallback Reason", surface_meta.get("fallback_reason") or "none", "shown when applicable"),
                        ("Warnings", warning_text or "none", "latest metadata warnings"),
                    ],
                ),
                (
                    "Fit Diagnostics",
                    [
                        ("Fit Preset", surface_meta.get("fit_filter_preset") or "Standard", "row eligibility"),
                        (
                            "Fit Rows",
                            f"{fmt_int(surface_meta.get('fit_eligible_count'))} in / {fmt_int(surface_meta.get('fit_excluded_count'))} out",
                            "included vs excluded",
                        ),
                        (
                            "Fit Filters",
                            (
                                f"spread <= {fmt_pct(surface_meta.get('fit_max_bid_ask_spread_pct'), 0)}, "
                                f"age <= {fmt_int(surface_meta.get('fit_max_quote_age_days'))}d"
                            ),
                            "surface fit",
                        ),
                        (
                            "SVI",
                            (
                                f"{fmt_int((surface_meta.get('fit_diagnostics') or {}).get('fitted_expiries'))} exp, "
                                f"RMSE {fmt_pct((surface_meta.get('fit_diagnostics') or {}).get('rmse'))}"
                            ),
                            "smile fit",
                        ),
                        (
                            "SSVI",
                            (
                                f"{fmt_int((surface_meta.get('global_fit_diagnostics') or {}).get('fitted_expiries'))} exp, "
                                f"RMSE {fmt_pct((surface_meta.get('global_fit_diagnostics') or {}).get('rmse'))}"
                            ),
                            f"constraints {(surface_meta.get('global_fit_diagnostics') or {}).get('constraints_passed')}",
                        ),
                    ],
                ),
                (
                    "Market Integrity",
                    [
                        ("Market", market_status.get("reason", "unknown"), market_status.get("session_state", "Unknown")),
                        (
                            "Delay / Cache",
                            (
                                f"{fmt_int(market_status.get('data_delay_minutes'))} min / "
                                f"{fmt_int(surface_meta.get('cache_age_seconds'))}s"
                            ),
                            "provider timing",
                        ),
                        (
                            "Quote Flags",
                            (
                                f"stale {fmt_int(surface_meta.get('stale_quote_count'))}, "
                                f"last-only {fmt_int(surface_meta.get('last_only_quote_count'))}"
                            ),
                            "row quality",
                        ),
                        (
                            "Static Checks",
                            (
                                f"crossed/locked {fmt_int(surface_meta.get('crossed_locked_rejected_count'))}, "
                                f"parity {fmt_int(surface_meta.get('parity_violation_count'))}, "
                                f"no-arb {fmt_int(surface_meta.get('no_arbitrage_violation_count'))}"
                            ),
                            f"excluded {fmt_int(surface_meta.get('no_arbitrage_excluded_count'))}",
                        ),
                        (
                            "Expected Move",
                            (
                                f"{fmt_money(surface_meta.get('front_expected_move'))} "
                                f"({fmt_pct(surface_meta.get('front_expected_move_pct'))})"
                            ),
                            surface_meta.get("front_expected_move_method") or "n/a",
                        ),
                    ],
                ),
                (
                    "Model Inputs",
                    [
                        ("IV Price Source", surface_meta.get("option_price_source") or "mark", "selected market price"),
                        ("Pricing Model", surface_meta.get("pricing_model_label") or "BSM with dividends", "contract analytics"),
                        (
                            "Rates",
                            f"{surface_meta.get('risk_free_rate_source') or 'unknown'} / {fmt_pct(surface_meta.get('risk_free_rate_30d'))}",
                            "30D risk-free",
                        ),
                        (
                            "Dividends",
                            (
                                f"{surface_meta.get('dividend_source') or 'unknown'} / "
                                f"{fmt_pct(surface_meta.get('effective_dividend_yield_30d'))}"
                            ),
                            "30D effective yield",
                        ),
                        (
                            "Model Counts",
                            (
                                f"Greeks {fmt_int(surface_meta.get('contract_greeks_count'))}, "
                                f"computed IV {fmt_int(surface_meta.get('computed_iv_count'))}"
                            ),
                            f"corp actions {fmt_int(surface_meta.get('corporate_action_warning_count'))}",
                        ),
                    ],
                ),
                (
                    "Surface Tape",
                    [
                        (
                            "Filter Guardrails",
                            (
                                f"OI >= {fmt_int(surface_meta.get('min_open_interest'))}, "
                                f"volume >= {fmt_int(surface_meta.get('min_volume'))}, "
                                f"spread <= {fmt_pct(surface_meta.get('max_bid_ask_spread_pct'), 0)}, "
                                f"age <= {fmt_int(surface_meta.get('max_quote_age_days'))}d"
                            ),
                            "chain display",
                        ),
                        (
                            "Fit Guardrails",
                            (
                                f"volume >= {fmt_int(surface_meta.get('fit_min_volume'))}, "
                                f"OI >= {fmt_int(surface_meta.get('fit_min_open_interest'))}, "
                                f"moneyness {surface_meta.get('fit_moneyness_min', 'n/a')}-"
                                f"{surface_meta.get('fit_moneyness_max', 'n/a')}"
                            ),
                            f"raw IV <= {fmt_pct(surface_meta.get('fit_max_raw_iv'), 0)}",
                        ),
                        (
                            "Fit Policies",
                            (
                                f"no-arb {surface_meta.get('fit_no_arbitrage_policy') or 'exclude'}, "
                                f"last-only {surface_meta.get('fit_last_only_policy') or 'allow_penalized'}"
                            ),
                            "diagnostic honesty",
                        ),
                        (
                            "Snapshot History",
                            (
                                f"rank {fmt_pct(surface_meta.get('iv_rank'))}, "
                                f"percentile {fmt_pct(surface_meta.get('iv_percentile'))}, "
                                f"history {fmt_int(surface_meta.get('iv_history_observations'))}"
                            ),
                            "local snapshots",
                        ),
                        (
                            "Tape Change",
                            (
                                f"points {fmt_int(surface_meta.get('surface_change_points'))}, "
                                f"tape {fmt_int(surface_meta.get('surface_tape_snapshots'))}, "
                                f"ATM dIV {fmt_pct(surface_meta.get('atm_iv_change'))}, "
                                f"vol-of-vol {fmt_pct(surface_meta.get('snapshot_vol_of_vol'))}"
                            ),
                            f"rich/cheap {fmt_int(surface_meta.get('rich_cheap_candidates'))}",
                        ),
                    ],
                ),
                (
                    "Research Fits",
                    [
                        (
                            "Forward / Discount",
                            (
                                f"{fmt_money(surface_meta.get('forward_price_median'))} / "
                                f"{fmt_decimal(surface_meta.get('discount_factor_median'), 5)}"
                            ),
                            "median inputs",
                        ),
                        (
                            "Heston",
                            (
                                f"{surface_meta.get('heston_research_status') or 'n/a'} / "
                                f"RMSE {fmt_pct(surface_meta.get('heston_research_rmse'))}"
                            ),
                            "research diagnostic",
                        ),
                        (
                            "SABR",
                            f"{surface_meta.get('sabr_status') or 'n/a'} / RMSE {fmt_pct(surface_meta.get('sabr_rmse'))}",
                            "research diagnostic",
                        ),
                    ],
                ),
            ],
            reason_buckets=reason_buckets,
            alert_level=quality_alert["level"],
            alert_message=quality_alert["message"],
        ),
        unsafe_allow_html=True,
    )

    def build_market_snapshot() -> pd.DataFrame:
        market_rows = []
        for symbol in selected_symbols:
            data = get_current_data_cached(symbol, data_key)
            market_rows.append(
                {
                    "Symbol": symbol,
                    "Spot": data.get("price"),
                    "30D IV": data.get("iv_30d"),
                    "60D IV": data.get("iv_60d"),
                    "90D IV": data.get("iv_90d"),
                    "30D Rate": data.get("risk_free_rate_30d"),
                    "30D Div Yield": data.get("dividend_yield_30d"),
                    "Action Warnings": data.get("corporate_action_warning_count"),
                    "Delta": data.get("delta"),
                    "Gamma": data.get("gamma"),
                    "Theta/day": data.get("theta"),
                    "Vega/1%": data.get("vega"),
                    "Contracts": data.get("contracts"),
                    "Volume": data.get("volume"),
                    "Mode": data.get("data_mode"),
                    "IV Source": data.get("iv_source"),
                }
            )
        return pd.DataFrame(market_rows)

    def _term_event_markers(term: pd.DataFrame, moves: list[dict[str, Any]], expiry_events: dict[str, Any]):
        if term.empty or not moves or not expiry_events:
            return []
        dte_by_expiry = {str(item.get("expiration")): item.get("dte") for item in moves}
        rows = []
        for expiry, events in expiry_events.items():
            dte = dte_by_expiry.get(str(expiry))
            if dte is None:
                continue
            nearest = term.iloc[(term["DTE"] - float(dte)).abs().idxmin()]
            labels = [
                f"{event.get('event_type', 'event')}: {event.get('event_date')} {event.get('description', '')}"
                for event in events or []
            ]
            rows.append({"dte": float(dte), "atm_iv": float(nearest["ATM IV"]), "label": " | ".join(labels[:4])})
        return rows

    def _event_rows(expiry_events: dict[str, Any]):
        rows = []
        for expiry, events in sorted((expiry_events or {}).items()):
            for event in events or []:
                rows.append(
                    {
                        "Expiry": expiry,
                        "Event Date": event.get("event_date"),
                        "Type": event.get("event_type"),
                        "Symbol": event.get("symbol"),
                        "Description": event.get("description"),
                        "Source": event.get("source"),
                    }
                )
        return rows

    def _fmt_number(value: Any, digits: int = 3) -> str:
        return "n/a" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"

    def _surface_points_frame(points: list[dict[str, Any]], value_col: str) -> pd.DataFrame:
        frame = pd.DataFrame(points or [])
        if frame.empty:
            return frame
        for column in ("strike", "dte", value_col):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["strike", "dte", value_col])

    def _fit_points_from_metadata(meta: dict[str, Any]) -> pd.DataFrame:
        rows = []
        ssvi_residuals = (meta.get("ssvi_surface") or {}).get("residuals") or []
        for row in ssvi_residuals:
            rows.append({**row, "source": "SSVI"})
        if not rows:
            for smile in meta.get("svi_smiles") or []:
                for row in smile.get("residuals") or []:
                    rows.append(
                        {
                            **row,
                            "dte": smile.get("dte"),
                            "expiration": smile.get("expiration"),
                            "source": "SVI",
                        }
                    )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        for column in ("strike", "dte", "log_moneyness", "observed_iv", "fitted_iv", "residual"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=["strike", "dte", "observed_iv", "fitted_iv", "residual"])

    def _fit_point_axis(frame: pd.DataFrame, axis: str, spot: float) -> tuple[pd.Series, str, str]:
        axis_key = str(axis or "strike").lower().replace(" ", "_").replace("-", "_")
        if axis_key in {"moneyness", "spot_moneyness"}:
            return frame["strike"] / spot, "Moneyness (K/S)", "Moneyness"
        if axis_key in {"log_moneyness", "logmoneyness"} and "log_moneyness" in frame:
            return frame["log_moneyness"], "Log-moneyness ln(K/S)", "Log-moneyness"
        if axis_key in {"delta", "call_delta"}:
            from src.pricing.black_scholes import OptionGreeks

            values = []
            for row in frame.itertuples():
                try:
                    values.append(
                        OptionGreeks.delta(
                            spot,
                            float(row.strike),
                            max(float(row.dte) / 365.0, 1e-8),
                            0.0,
                            max(float(row.fitted_iv), 1e-8),
                            "call",
                        )
                    )
                except (TypeError, ValueError, AttributeError):
                    values.append(np.nan)
            return pd.Series(values, index=frame.index), "Call delta", "Call delta"
        return frame["strike"], "Strike", "Strike"

    def _analysis_export_payload() -> dict[str, Any]:
        data_timestamp = surface_meta.get("timestamp") or surface_meta.get("spot_timestamp") or current_data.get("timestamp")
        if hasattr(data_timestamp, "isoformat"):
            data_timestamp = data_timestamp.isoformat()
        return {
            "symbol": surface_symbol,
            "spot": current_data.get("price"),
            "data_timestamp": str(data_timestamp or datetime.now().isoformat()),
            "model_assumptions": surface_meta.get("pricing_model_label") or current_data.get("pricing_model_label"),
            "surface_summary": {
                "atm_iv": stats.get("atm_iv"),
                "iv_rank": surface_meta.get("iv_rank"),
                "iv_percentile": surface_meta.get("iv_percentile"),
                "surface_points": stats.get("points"),
                "term_spread": stats.get("term_spread"),
            },
            "diagnostics": {
                "surface_quality_score": surface_meta.get("surface_quality_score"),
                "fit_diagnostics": surface_meta.get("fit_diagnostics"),
                "global_fit_diagnostics": surface_meta.get("global_fit_diagnostics"),
                "fit_mode_comparison": surface_meta.get("fit_mode_comparison"),
                "post_fit_arbitrage": surface_meta.get("post_fit_arbitrage"),
                "surface_repair": surface_meta.get("surface_repair"),
                "quality_drop_alert": surface_meta.get("quality_drop_alert"),
                "warnings": surface_meta.get("warnings"),
                "source": surface_meta.get("surface_source") or current_data.get("price_source"),
                "mode": surface_meta.get("surface_mode") or current_data.get("data_mode"),
            },
            "provenance": {
                "surface_source": surface_meta.get("surface_source"),
                "surface_mode": surface_meta.get("surface_mode"),
                "option_price_source": surface_meta.get("option_price_source"),
                "pricing_model": surface_meta.get("pricing_model_label"),
                "quality_score": surface_meta.get("surface_quality_score"),
            },
        }

    def _heatmap_from_points(points: list[dict[str, Any]], value_col: str) -> pd.DataFrame:
        frame = _surface_points_frame(points, value_col)
        if frame.empty:
            return frame
        return frame.pivot_table(index="dte", columns="strike", values=value_col, aggfunc="mean").sort_index()

    def _surface_replay_figure(tape_payload: dict[str, Any]) -> go.Figure | None:
        snapshots = tape_payload.get("snapshots") or []
        frames = []
        first_grid = None
        first_timestamp = None
        for snapshot in snapshots:
            grid = _heatmap_from_points(snapshot.get("points") or [], "iv")
            if grid.empty:
                continue
            timestamp = snapshot.get("timestamp", "snapshot")
            if first_grid is None:
                first_grid = grid
                first_timestamp = timestamp
            frames.append(
                go.Frame(
                    name=timestamp,
                    data=[
                        go.Heatmap(
                            z=grid.values,
                            x=list(grid.columns),
                            y=list(grid.index),
                            colorscale="Cividis",
                            zmin=None,
                            zmax=None,
                            colorbar=dict(title="IV"),
                        )
                    ],
                )
            )
        if first_grid is None:
            return None
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=first_grid.values,
                    x=list(first_grid.columns),
                    y=list(first_grid.index),
                    colorscale="Cividis",
                    colorbar=dict(title="IV"),
                    hovertemplate="Strike: %{x:.2f}<br>DTE: %{y:.0f}<br>IV: %{z:.2%}<extra></extra>",
                )
            ],
            frames=frames,
        )
        steps = [
            {
                "method": "animate",
                "label": frame.name[-8:] if len(frame.name) >= 8 else frame.name,
                "args": [[frame.name], {"mode": "immediate", "frame": {"duration": 250, "redraw": True}}],
            }
            for frame in frames
        ]
        fig.update_layout(
            title=f"{surface_symbol} Surface Tape Replay",
            xaxis_title="Strike",
            yaxis_title="Days to expiry",
            sliders=[{"active": 0, "currentvalue": {"prefix": "Timestamp "}, "steps": steps}],
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 450, "redraw": True}, "fromcurrent": True}],
                        }
                    ],
                }
            ],
        )
        fig.add_annotation(
            text=str(first_timestamp),
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.08,
            showarrow=False,
            font=dict(size=11, color="#667085"),
        )
        return fig

    def _prior_comparison_figure(
        records: list[dict[str, Any]],
        value_col: str,
        title: str,
        colorbar_title: str,
        *,
        colorscale: str = "Cividis",
        zmid: float | None = None,
    ) -> go.Figure | None:
        grid = _heatmap_from_points(records, value_col)
        if grid.empty:
            return None
        heatmap_kwargs = {
            "z": grid.values,
            "x": list(grid.columns),
            "y": list(grid.index),
            "colorscale": colorscale,
            "colorbar": dict(title=colorbar_title),
            "hovertemplate": (
                "Strike: %{x:.2f}<br>DTE: %{y:.0f}<br>"
                f"{colorbar_title}: %{{z:.2%}}<extra></extra>"
            ),
        }
        if zmid is not None:
            heatmap_kwargs["zmid"] = zmid
        fig = go.Figure(data=[go.Heatmap(**heatmap_kwargs)])
        fig.update_layout(title=title, xaxis_title="Strike", yaxis_title="Days to expiry")
        return fig

    market_df = load_with_status(
        st,
        LoadingState(
            title="Universe price grid",
            detail="Fetching yfinance prices and cached chain summaries for the selected symbols.",
            stage="price grid",
            rows=max(3, min(len(selected_symbols), 8)),
        ),
        build_market_snapshot,
    )
    st.markdown(
        '<div class="dashboard-ready-marker" data-dashboard-state="ready">dashboard ready</div>',
        unsafe_allow_html=True,
    )

    page_registry = default_page_registry()
    tab_labels = function_key_page_titles(page_registry)
    render_keyboard_layer(page_labels=tab_labels, symbols=selected_symbols)
    (
        surface_tab,
        chain_tab,
        skew_tab,
        term_tab,
        quality_tab,
        scanner_tab,
        strategy_tab,
        risk_tab,
        diagnostics_tab,
        report_tab,
    ) = st.tabs(tab_labels)
    local_vol_tab = surface_tab
    relative_tab = scanner_tab

    with surface_tab:
        st.markdown('<div class="section-header">Volatility Surface</div>', unsafe_allow_html=True)
        axis_mesh, axis_expiry_mesh, axis_vols, axis_title, _axis_hover_format, axis_hover_label = surface_axis(
            strikes,
            expiries,
            vol_surface,
            current_data["price"],
            surface_x_axis,
            surface_meta.get("surface_risk_free_rate") or surface_meta.get("risk_free_rate_median"),
            surface_meta.get("surface_dividend_yield") or surface_meta.get("effective_dividend_yield_median"),
        )
        fit_points = _fit_points_from_metadata(surface_meta)
        if not fit_points.empty:
            fit_points = fit_points.copy()
            fit_points["axis_value"], fit_axis_title, fit_axis_label = _fit_point_axis(
                fit_points,
                surface_x_axis,
                current_data["price"],
            )
            fit_points["reliability_overlay"] = pd.to_numeric(
                fit_points.get("fit_weight", 1.0),
                errors="coerce",
            ).fillna(1.0).clip(0.0, 1.0)
        else:
            fit_axis_title = axis_title
            fit_axis_label = axis_title
        if fit_mode_view.get("warning"):
            st.caption(fit_mode_view["warning"])

        if show_3d_surface:
            fig_3d = go.Figure(
                data=[
                    go.Surface(
                        z=axis_vols,
                        x=axis_mesh,
                        y=axis_expiry_mesh,
                        colorscale="Cividis",
                        colorbar=dict(title="IV"),
                        hovertemplate=f"{axis_hover_label}<br>DTE: %{{y:.0f}}<br>IV: %{{z:.2%}}<extra></extra>",
                    )
                ]
            )
            if not fit_points.empty:
                marker_3d = {"size": 3, "color": "#f97316", "opacity": 0.82}
                if show_reliability_overlay:
                    marker_3d = {
                        "size": 4 + (fit_points["reliability_overlay"] * 5),
                        "color": fit_points["reliability_overlay"],
                        "colorscale": "Viridis",
                        "cmin": 0,
                        "cmax": 1,
                        "colorbar": dict(title="Reliability"),
                        "opacity": 0.82,
                    }
                fig_3d.add_trace(
                    go.Scatter3d(
                        x=fit_points["axis_value"],
                        y=fit_points["dte"],
                        z=fit_points["observed_iv"],
                        mode="markers",
                        name="Raw IV points",
                        marker=marker_3d,
                        customdata=fit_points[["fitted_iv", "residual", "source", "reliability_overlay"]],
                        hovertemplate=(
                            f"{fit_axis_label}: %{{x:.3f}}<br>DTE: %{{y:.0f}}<br>"
                            "Observed IV: %{z:.2%}<br>Fitted IV: %{customdata[0]:.2%}<br>"
                            "Residual: %{customdata[1]:.2%}<br>Reliability: %{customdata[3]:.2f}<br>"
                            "%{customdata[2]}<extra></extra>"
                        ),
                    )
                )
            fig_3d.update_layout(
                title=f"{surface_symbol} Implied Volatility Surface - {fit_mode_view['chart_label']}",
                scene=dict(
                    xaxis_title=fit_axis_title,
                    yaxis_title="Days to expiry",
                    zaxis_title="Annualized IV",
                    camera=dict(eye=dict(x=1.45, y=1.35, z=1.0)),
                ),
                margin=dict(l=0, r=0, t=45, b=0),
                height=620,
            )
            with card(st, title="Implied Volatility Surface", kicker="Surface", actions=["R", "EXP", "i"]):
                st.plotly_chart(apply_chart_layout(fig_3d, 620), width="stretch")

        fig_heatmap = go.Figure(
            data=[
                go.Heatmap(
                    z=axis_vols,
                    x=axis_mesh[0, :] if axis_mesh.ndim == 2 else axis_mesh,
                    y=axis_expiry_mesh[:, 0] if axis_expiry_mesh.ndim == 2 else axis_expiry_mesh,
                    colorscale="RdBu_r",
                    colorbar=dict(title="IV"),
                    hovertemplate=f"{axis_hover_label}<br>DTE: %{{y:.0f}}<br>IV: %{{z:.2%}}<extra></extra>",
                )
            ]
        )
        if not fit_points.empty:
            marker_2d = {"color": "#111827", "size": 6, "symbol": "circle-open"}
            if show_reliability_overlay:
                marker_2d = {
                    "color": fit_points["reliability_overlay"],
                    "size": 5 + (fit_points["reliability_overlay"] * 8),
                    "symbol": "circle-open",
                    "colorscale": "Viridis",
                    "cmin": 0,
                    "cmax": 1,
                    "colorbar": dict(title="Reliability"),
                }
            fig_heatmap.add_trace(
                go.Scatter(
                    x=fit_points["axis_value"],
                    y=fit_points["dte"],
                    mode="markers",
                    name="Raw IV points",
                    marker=marker_2d,
                    customdata=fit_points[["observed_iv", "fitted_iv", "residual", "source", "reliability_overlay"]],
                    hovertemplate=(
                        f"{fit_axis_label}: %{{x:.3f}}<br>DTE: %{{y:.0f}}<br>"
                        "Observed IV: %{customdata[0]:.2%}<br>"
                        "Fitted IV: %{customdata[1]:.2%}<br>"
                        "Residual: %{customdata[2]:.2%}<br>Reliability: %{customdata[4]:.2f}<br>"
                        "%{customdata[3]}<extra></extra>"
                    ),
                )
            )
        fig_heatmap.update_layout(
            title=f"{surface_symbol} Surface Heatmap - {fit_mode_view['chart_label']}",
            xaxis_title=axis_title,
            yaxis_title="Days to expiry",
        )
        with card(st, title="Surface Heatmap", kicker="Surface", actions=["EXP", "i"]):
            st.plotly_chart(apply_chart_layout(fig_heatmap, 430), width="stretch")
            st.caption(
                f"Selected fit view {fit_mode_view['selected_mode']}; "
                f"provenance {display_provenance_label(fit_mode_view['provenance'])}. "
                "Prior-assisted, ML-denoised, repaired, and diagnostic raw values are estimates or overlays, not market observations."
            )

        prior_comparison = surface_meta.get("surface_prior_comparison") or []
        if prior_comparison:
            st.markdown('<div class="section-header">Historical Prior Comparison</div>', unsafe_allow_html=True)
            prior_cols = st.columns(2)
            current_prior_fig = _prior_comparison_figure(
                prior_comparison,
                "current_iv",
                f"{surface_symbol} Current Robust Fit Estimate",
                "Current IV",
            )
            historical_prior_fig = _prior_comparison_figure(
                prior_comparison,
                "prior_iv",
                f"{surface_symbol} Historical Prior Estimate",
                "Prior IV",
            )
            with prior_cols[0]:
                if current_prior_fig is not None:
                    st.plotly_chart(apply_chart_layout(current_prior_fig, 360), width="stretch")
            with prior_cols[1]:
                if historical_prior_fig is not None:
                    st.plotly_chart(apply_chart_layout(historical_prior_fig, 360), width="stretch")
            prior_change_fig = _prior_comparison_figure(
                prior_comparison,
                "iv_change",
                f"{surface_symbol} Current Minus Historical Prior",
                "dIV",
                colorscale="RdBu",
                zmid=0.0,
            )
            if prior_change_fig is not None:
                st.plotly_chart(apply_chart_layout(prior_change_fig, 390), width="stretch")
            st.caption(
                "Historical prior and prior-assisted surface values are estimates, not market observations. "
                f"Prior source {surface_meta.get('surface_prior_source') or 'persisted snapshots'}; "
                f"age {_fmt_number(surface_meta.get('surface_prior_age_days'), 2)} days; "
                f"overlap {fmt_int(surface_meta.get('surface_prior_overlap_count'))}; "
                f"blend weight {fmt_pct(surface_meta.get('surface_prior_blend_weight'))}; "
                f"applied {surface_meta.get('surface_prior_applied', False)}."
            )
        else:
            prior_meta = surface_meta.get("historical_surface_prior") or {}
            st.markdown(
                render_empty_state(
                    "Historical prior comparison unavailable",
                    prior_meta.get("reason")
                    or "No recent persisted prior grid overlaps the current fitted surface.",
                    "Store recent overlapping snapshots to compare current fit estimates with historical priors.",
                ),
                unsafe_allow_html=True,
            )

        if not fit_points.empty:
            residual_fig = go.Figure(
                data=[
                    go.Scatter(
                        x=fit_points["axis_value"],
                        y=fit_points["residual"],
                        mode="markers",
                        marker=dict(
                            color=fit_points["dte"],
                            colorscale="Cividis",
                            colorbar=dict(title="DTE"),
                            size=8,
                        ),
                        customdata=fit_points[["dte", "observed_iv", "fitted_iv", "source"]],
                        hovertemplate=(
                            f"{fit_axis_label}: %{{x:.3f}}<br>DTE: %{{customdata[0]:.0f}}<br>"
                            "Residual: %{y:.2%}<br>Observed IV: %{customdata[1]:.2%}<br>"
                            "Fitted IV: %{customdata[2]:.2%}<br>%{customdata[3]}<extra></extra>"
                        ),
                    )
                ]
            )
            residual_fig.add_hline(y=0, line_width=1, line_color="#667085")
            residual_fig.update_layout(
                title=f"{surface_symbol} Fit Residuals",
                xaxis_title=fit_axis_title,
                yaxis_title="Fitted minus raw IV",
            )
            st.plotly_chart(apply_chart_layout(residual_fig, 340), width="stretch")
            st.caption(
                f"Raw/fitted overlay source: {fit_points['source'].iloc[0]}; "
                f"residual points {fmt_int(len(fit_points))}; "
                f"surface source {surface_meta.get('surface_source', 'unknown')}; "
                f"mode {surface_meta.get('surface_mode', 'unknown')}."
            )
        else:
            st.markdown(
                render_empty_state(
                    "Fit residuals unavailable",
                    "No SVI or SSVI residual payload is available for the current fitted surface.",
                    "Use a symbol with enough valid IV rows for surface calibration.",
                ),
                unsafe_allow_html=True,
            )

        tape_payload = surface_meta.get("surface_tape") or {}
        replay_fig = _surface_replay_figure(tape_payload)
        if replay_fig is not None:
            st.plotly_chart(apply_chart_layout(replay_fig, 450), width="stretch")
            st.caption(
                f"Replay source: {tape_payload.get('source', 'persisted snapshots')}; "
                f"snapshots: {fmt_int(tape_payload.get('snapshot_count'))}."
            )
        else:
            st.markdown(
                render_empty_state(
                    "Surface tape unavailable",
                    "No same-day persisted snapshots have usable IV rows for replay.",
                    "Refresh during the session to add timestamped local snapshots.",
                ),
                unsafe_allow_html=True,
            )

        heatmaps = (surface_meta.get("surface_change_heatmaps") or {}).get("baselines") or {}
        available_heatmap_keys = [key for key, payload in heatmaps.items() if payload.get("available")]
        if available_heatmap_keys:
            selected_baseline = st.selectbox(
                "Surface change baseline",
                available_heatmap_keys,
                format_func=lambda key: heatmaps[key].get("label", key.replace("_", " ").title()),
                help="Baseline used for the current-minus-baseline IV heatmap.",
            )
            selected_heatmap = heatmaps[selected_baseline]
            change_grid = _heatmap_from_points(selected_heatmap.get("records") or [], "iv_change")
            if not change_grid.empty:
                fig_change_heatmap = go.Figure(
                    data=[
                        go.Heatmap(
                            z=change_grid.values,
                            x=list(change_grid.columns),
                            y=list(change_grid.index),
                            colorscale="RdBu",
                            zmid=0,
                            colorbar=dict(title="dIV"),
                            hovertemplate=(
                                "Strike: %{x:.2f}<br>DTE: %{y:.0f}<br>"
                                "Current - baseline IV: %{z:.2%}<extra></extra>"
                            ),
                        )
                    ]
                )
                fig_change_heatmap.update_layout(
                    title=f"{surface_symbol} Surface Change Heatmap",
                    xaxis_title="Strike",
                    yaxis_title="Days to expiry",
                )
                st.plotly_chart(apply_chart_layout(fig_change_heatmap, 430), width="stretch")
                st.caption(
                    f"Baseline {selected_heatmap.get('baseline_timestamp', 'unknown')} "
                    f"({selected_heatmap.get('baseline_mode', 'unknown')}); "
                    f"matched points {fmt_int(selected_heatmap.get('matched_points'))}; "
                    f"mean dIV {fmt_pct(selected_heatmap.get('mean_iv_change'))}; "
                    f"max absolute dIV {fmt_pct(selected_heatmap.get('max_abs_iv_change'))}."
                )
        else:
            st.markdown(
                render_empty_state(
                    "Surface change heatmap unavailable",
                    "No previous refresh, previous hour, or previous close baseline has matching IV rows.",
                    "Store at least two snapshots with overlapping strikes and expiries.",
                ),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Market Snapshot</div>', unsafe_allow_html=True)
        market_filter_cols = st.columns([1, 1, 1])
        mode_options = sorted(str(mode) for mode in market_df["Mode"].dropna().unique()) if "Mode" in market_df else []
        with market_filter_cols[0]:
            market_modes = st.multiselect(
                "Market mode",
                mode_options,
                default=mode_options,
                help="Filter market rows by data provenance mode.",
            )
        with market_filter_cols[1]:
            min_market_iv = st.slider(
                "Min 30D IV",
                0.0,
                2.0,
                0.0,
                0.05,
                format="%.2f",
                help="Minimum displayed 30D annualized implied volatility.",
            )
        market_display = filter_market_snapshot(market_df, market_modes, min_market_iv)
        with market_filter_cols[2]:
            st.download_button(
                "Export market CSV",
                dataframe_to_csv_bytes(market_display),
                file_name=f"{surface_symbol}_market_snapshot.csv",
                mime="text/csv",
                width="stretch",
            )
        st.dataframe(
            market_display,
            width="stretch",
            hide_index=True,
            column_config={
                "Spot": st.column_config.NumberColumn(format="$%.2f", help=COLUMN_HELP["Spot"]),
                "30D IV": st.column_config.NumberColumn(format="%.2%", help=COLUMN_HELP["30D IV"]),
                "60D IV": st.column_config.NumberColumn(format="%.2%", help=COLUMN_HELP["60D IV"]),
                "90D IV": st.column_config.NumberColumn(format="%.2%", help=COLUMN_HELP["90D IV"]),
                "30D Rate": st.column_config.NumberColumn(format="%.2%", help=COLUMN_HELP["30D Rate"]),
                "30D Div Yield": st.column_config.NumberColumn(
                    format="%.2%",
                    help=COLUMN_HELP["30D Div Yield"],
                ),
                "Action Warnings": st.column_config.NumberColumn(
                    format="%d",
                    help=COLUMN_HELP["Action Warnings"],
                ),
                "Delta": st.column_config.NumberColumn(format="%.4f", help=COLUMN_HELP["Delta"]),
                "Gamma": st.column_config.NumberColumn(format="%.4f", help=COLUMN_HELP["Gamma"]),
                "Theta/day": st.column_config.NumberColumn(format="$%.4f", help=COLUMN_HELP["Theta/day"]),
                "Vega/1%": st.column_config.NumberColumn(format="$%.4f", help=COLUMN_HELP["Vega/1%"]),
                "Contracts": st.column_config.NumberColumn(format="%d", help=COLUMN_HELP["Contracts"]),
                "Volume": st.column_config.NumberColumn(format="%d", help=COLUMN_HELP["Volume"]),
                "Mode": st.column_config.TextColumn(help=COLUMN_HELP["Mode"]),
                "IV Source": st.column_config.TextColumn(help=COLUMN_HELP["IV Source"]),
            },
        )

    with chain_tab:
        st.markdown('<div class="section-header">Option Chain Explorer</div>', unsafe_allow_html=True)
        market_snapshot = load_with_status(
            st,
            LoadingState(
                title=f"{surface_symbol} market snapshot",
                detail="Fetching canonical price, option quotes, expirations, and provenance metadata.",
                stage="snapshot",
                rows=8,
            ),
            lambda: get_market_snapshot_cached(surface_symbol, data_key),
        )
        chain_df = market_snapshot.options_frame()
        chain_meta = market_snapshot.metadata_dict()
        if show_chain and not chain_df.empty:
            filter_cols = st.columns([1, 1, 1, 1])
            type_options = sorted(str(value) for value in chain_df["type"].dropna().unique()) if "type" in chain_df else []
            expiration_options = (
                sorted(pd.to_datetime(chain_df["expiration"], errors="coerce").dropna().dt.date.unique())
                if "expiration" in chain_df
                else []
            )
            with filter_cols[0]:
                selected_types = st.multiselect(
                    "Type",
                    type_options,
                    default=type_options,
                    help="Filter option-chain rows by call or put.",
                )
            with filter_cols[1]:
                selected_expirations = st.multiselect(
                    "Expiry",
                    expiration_options,
                    default=expiration_options[: min(4, len(expiration_options))],
                    help="Filter option-chain rows by expiration date.",
                )
            with filter_cols[2]:
                moneyness_band = st.slider(
                    "Moneyness",
                    0.35,
                    2.50,
                    (0.80, 1.20),
                    0.05,
                    help="Strike divided by spot.",
                )
            with filter_cols[3]:
                iv_band = st.slider(
                    "IV",
                    0.00,
                    5.00,
                    (0.05, 2.00),
                    0.05,
                    format="%.2f",
                    help="Annualized implied-volatility filter.",
                )
            show_advanced_greeks = st.checkbox(
                "Advanced Greeks",
                value=False,
                help=CONTROL_HELP["show_advanced_greeks"],
            )

            filtered = filter_option_chain(
                chain_df,
                max_spread_pct=max_spread_pct,
                min_open_interest=min_open_interest,
                min_volume=min_volume,
                max_quote_age_days=max_quote_age_days,
                option_types=selected_types,
                expirations=selected_expirations,
                moneyness_range=moneyness_band,
                iv_range=iv_band,
            )

            display_cols = [
                "type",
                "expiration",
                "daysToExpiration",
                "strike",
                "moneyness",
                "forwardMoneyness",
                "logMoneyness",
                "bid",
                "ask",
                "mid",
                "mark",
                "last",
                "selectedMarketPrice",
                "selectedPriceSource",
                "volume",
                "openInterest",
                "impliedVolatility",
                "computedIV",
                "intrinsicValue",
                "timeValue",
                "carryValue",
                "impliedVolContribution",
                "modelResidual",
                "pricingModel",
                "selectedModelPrice",
                "selectedModelResidual",
                "delta",
                "gamma",
                "theta",
                "vega",
                "rho",
                "europeanPrice",
                "americanPrice",
                "earlyExercisePremium",
                "earlyExerciseFlag",
                "parityViolation",
                "parityError",
                "noArbitrageViolation",
                "noArbitrageReasons",
                "riskFreeRate",
                "discountFactor",
                "forwardPrice",
                "effectiveDividendYield",
                "discreteDividendAmount",
                "quoteQuality",
                "isCrossedMarket",
                "isLockedMarket",
                "quoteAgeSeconds",
                "bidAskSpreadPct",
                "displayEligible",
                "displayRejectionReasons",
                "quoteReliabilityScore",
                "fitWeight",
                "fitEligible",
                "fitPenaltyReasons",
                "fitHardRejectionReasons",
            ]
            display_cols = [col for col in display_cols if col in filtered.columns]
            chain_display = add_freshness_column(
                filtered[display_cols],
                max_quote_age_days=max_quote_age_days,
            )
            st.download_button(
                "Export chain CSV",
                dataframe_to_csv_bytes(filtered[display_cols]),
                file_name=f"{surface_symbol}_option_chain.csv",
                mime="text/csv",
            )
            st.dataframe(
                chain_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "Freshness": st.column_config.TextColumn("Fresh", help="OK when quote age is within the display threshold; STALE otherwise."),
                    "type": st.column_config.TextColumn("Type", help=COLUMN_HELP["type"]),
                    "expiration": st.column_config.DateColumn("Expiration", help=COLUMN_HELP["expiration"]),
                    "daysToExpiration": st.column_config.NumberColumn(
                        "DTE",
                        format="%d",
                        help=COLUMN_HELP["daysToExpiration"],
                    ),
                    "strike": st.column_config.NumberColumn(format="$%.2f", help=COLUMN_HELP["strike"]),
                    "moneyness": st.column_config.NumberColumn(format="%.3f", help=COLUMN_HELP["moneyness"]),
                    "forwardMoneyness": st.column_config.NumberColumn(
                        "Fwd Mny",
                        format="%.3f",
                        help=COLUMN_HELP["forwardMoneyness"],
                    ),
                    "logMoneyness": st.column_config.NumberColumn(
                        "Log Mny",
                        format="%.3f",
                        help=COLUMN_HELP["logMoneyness"],
                    ),
                    "bid": st.column_config.NumberColumn(format="$%.2f", help=COLUMN_HELP["bid"]),
                    "ask": st.column_config.NumberColumn(format="$%.2f", help=COLUMN_HELP["ask"]),
                    "mid": st.column_config.NumberColumn(format="$%.2f", help=COLUMN_HELP["mid"]),
                    "mark": st.column_config.NumberColumn(format="$%.2f", help=COLUMN_HELP["mark"]),
                    "last": st.column_config.NumberColumn(format="$%.2f", help=COLUMN_HELP["last"]),
                    "selectedMarketPrice": st.column_config.NumberColumn(
                        "IV Price",
                        format="$%.2f",
                        help=COLUMN_HELP["selectedMarketPrice"],
                    ),
                    "selectedPriceSource": st.column_config.TextColumn(
                        "IV Price Source",
                        help=COLUMN_HELP["selectedPriceSource"],
                    ),
                    "volume": st.column_config.NumberColumn(format="%d", help=COLUMN_HELP["Volume"]),
                    "openInterest": st.column_config.NumberColumn(format="%d", help=COLUMN_HELP["openInterest"]),
                    "impliedVolatility": st.column_config.NumberColumn(
                        "Provider IV",
                        format="%.2%",
                        help=COLUMN_HELP["impliedVolatility"],
                    ),
                    "computedIV": st.column_config.NumberColumn(
                        "Computed IV",
                        format="%.2%",
                        help=COLUMN_HELP["computedIV"],
                    ),
                    "intrinsicValue": st.column_config.NumberColumn(
                        "Intrinsic",
                        format="$%.2f",
                        help=COLUMN_HELP["intrinsicValue"],
                    ),
                    "timeValue": st.column_config.NumberColumn(
                        "Time Value",
                        format="$%.2f",
                        help=COLUMN_HELP["timeValue"],
                    ),
                    "carryValue": st.column_config.NumberColumn(
                        "Carry",
                        format="$%.2f",
                        help=COLUMN_HELP["carryValue"],
                    ),
                    "impliedVolContribution": st.column_config.NumberColumn(
                        "IV Value",
                        format="$%.2f",
                        help=COLUMN_HELP["impliedVolContribution"],
                    ),
                    "modelResidual": st.column_config.NumberColumn(
                        "Model Residual",
                        format="$%.2f",
                        help=COLUMN_HELP["modelResidual"],
                    ),
                    "pricingModel": st.column_config.TextColumn(
                        "Model",
                        help=COLUMN_HELP["pricingModel"],
                    ),
                    "selectedModelPrice": st.column_config.NumberColumn(
                        "Model Price",
                        format="$%.2f",
                        help=COLUMN_HELP["selectedModelPrice"],
                    ),
                    "selectedModelResidual": st.column_config.NumberColumn(
                        "Model Residual",
                        format="$%.2f",
                        help=COLUMN_HELP["selectedModelResidual"],
                    ),
                    "delta": st.column_config.NumberColumn("Delta", format="%.4f", help=COLUMN_HELP["Delta"]),
                    "gamma": st.column_config.NumberColumn("Gamma", format="%.4f", help=COLUMN_HELP["Gamma"]),
                    "theta": st.column_config.NumberColumn("Theta/day", format="$%.4f", help=COLUMN_HELP["Theta/day"]),
                    "vega": st.column_config.NumberColumn("Vega/1%", format="$%.4f", help=COLUMN_HELP["Vega/1%"]),
                    "rho": st.column_config.NumberColumn("Rho/1%", format="$%.4f", help=COLUMN_HELP["rho"]),
                    "europeanPrice": st.column_config.NumberColumn(
                        "European",
                        format="$%.2f",
                        help=COLUMN_HELP["europeanPrice"],
                    ),
                    "americanPrice": st.column_config.NumberColumn(
                        "American",
                        format="$%.2f",
                        help=COLUMN_HELP["americanPrice"],
                    ),
                    "earlyExercisePremium": st.column_config.NumberColumn(
                        "Early Ex Prem",
                        format="$%.2f",
                        help=COLUMN_HELP["earlyExercisePremium"],
                    ),
                    "earlyExerciseFlag": st.column_config.CheckboxColumn(
                        "Early Ex",
                        help=COLUMN_HELP["earlyExerciseFlag"],
                    ),
                    "parityViolation": st.column_config.CheckboxColumn(
                        "Parity Flag",
                        help=COLUMN_HELP["parityViolation"],
                    ),
                    "parityError": st.column_config.NumberColumn(
                        "Parity Error",
                        format="$%.2f",
                        help=COLUMN_HELP["parityError"],
                    ),
                    "noArbitrageViolation": st.column_config.CheckboxColumn(
                        "No-Arb Flag",
                        help=COLUMN_HELP["noArbitrageViolation"],
                    ),
                    "noArbitrageReasons": st.column_config.TextColumn(
                        "No-Arb Reasons",
                        help=COLUMN_HELP["noArbitrageReasons"],
                    ),
                    "riskFreeRate": st.column_config.NumberColumn(
                        "Rate",
                        format="%.2%",
                        help=COLUMN_HELP["riskFreeRate"],
                    ),
                    "discountFactor": st.column_config.NumberColumn(
                        "DF",
                        format="%.5f",
                        help=COLUMN_HELP["discountFactor"],
                    ),
                    "forwardPrice": st.column_config.NumberColumn(
                        "Forward",
                        format="$%.2f",
                        help=COLUMN_HELP["forwardPrice"],
                    ),
                    "effectiveDividendYield": st.column_config.NumberColumn(
                        "Div Yield",
                        format="%.2%",
                        help=COLUMN_HELP["effectiveDividendYield"],
                    ),
                    "discreteDividendAmount": st.column_config.NumberColumn(
                        "Div $",
                        format="$%.2f",
                        help=COLUMN_HELP["discreteDividendAmount"],
                    ),
                    "quoteQuality": st.column_config.TextColumn(
                        "Quote Quality",
                        help=COLUMN_HELP["quoteQuality"],
                    ),
                    "isCrossedMarket": st.column_config.CheckboxColumn(
                        "Crossed",
                        help=COLUMN_HELP["isCrossedMarket"],
                    ),
                    "isLockedMarket": st.column_config.CheckboxColumn(
                        "Locked",
                        help=COLUMN_HELP["isLockedMarket"],
                    ),
                    "quoteAgeSeconds": st.column_config.NumberColumn(
                        "Quote Age Sec",
                        format="%.0f",
                        help=COLUMN_HELP["quoteAgeSeconds"],
                    ),
                    "bidAskSpreadPct": st.column_config.NumberColumn(
                        "Spread %",
                        format="%.2%",
                        help=COLUMN_HELP["bidAskSpreadPct"],
                    ),
                    "displayEligible": st.column_config.CheckboxColumn(
                        "Display Eligible",
                        help=COLUMN_HELP["displayEligible"],
                    ),
                    "displayRejectionReasons": st.column_config.TextColumn(
                        "Display Reasons",
                        help=COLUMN_HELP["displayRejectionReasons"],
                    ),
                    "quoteReliabilityScore": st.column_config.NumberColumn(
                        "Reliability",
                        format="%.2f",
                        help=COLUMN_HELP["quoteReliabilityScore"],
                    ),
                    "fitWeight": st.column_config.NumberColumn(
                        "Fit Weight",
                        format="%.2f",
                        help=COLUMN_HELP["fitWeight"],
                    ),
                    "fitEligible": st.column_config.CheckboxColumn(
                        "Fit Eligible",
                        help=COLUMN_HELP["fitEligible"],
                    ),
                    "fitPenaltyReasons": st.column_config.TextColumn(
                        "Fit Penalties",
                        help=COLUMN_HELP["fitPenaltyReasons"],
                    ),
                    "fitHardRejectionReasons": st.column_config.TextColumn(
                        "Fit Reject Reasons",
                        help=COLUMN_HELP["fitHardRejectionReasons"],
                    ),
                },
            )
            if show_advanced_greeks:
                advanced_cols = [
                    "contractSymbol",
                    "type",
                    "expiration",
                    "daysToExpiration",
                    "strike",
                    "computedIV",
                    "delta",
                    "gamma",
                    "vega",
                    "vanna",
                    "volga",
                    "vomma",
                    "charm",
                    "speed",
                    "color",
                ]
                advanced_cols = [col for col in advanced_cols if col in filtered.columns]
                if advanced_cols:
                    (advanced_tab,) = st.tabs(["Advanced Greeks"])
                    with advanced_tab:
                        st.dataframe(
                            filtered[advanced_cols],
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "contractSymbol": st.column_config.TextColumn("Contract"),
                                "type": st.column_config.TextColumn("Type", help=COLUMN_HELP["type"]),
                                "expiration": st.column_config.DateColumn(
                                    "Expiration",
                                    help=COLUMN_HELP["expiration"],
                                ),
                                "daysToExpiration": st.column_config.NumberColumn(
                                    "DTE",
                                    format="%d",
                                    help=COLUMN_HELP["daysToExpiration"],
                                ),
                                "strike": st.column_config.NumberColumn(
                                    "Strike",
                                    format="$%.2f",
                                    help=COLUMN_HELP["strike"],
                                ),
                                "computedIV": st.column_config.NumberColumn(
                                    "Computed IV",
                                    format="%.2%",
                                    help=COLUMN_HELP["computedIV"],
                                ),
                                "delta": st.column_config.NumberColumn(
                                    "Delta",
                                    format="%.4f",
                                    help=COLUMN_HELP["Delta"],
                                ),
                                "gamma": st.column_config.NumberColumn(
                                    "Gamma",
                                    format="%.4f",
                                    help=COLUMN_HELP["Gamma"],
                                ),
                                "vega": st.column_config.NumberColumn(
                                    "Vega/1%",
                                    format="$%.4f",
                                    help=COLUMN_HELP["Vega/1%"],
                                ),
                                "vanna": st.column_config.NumberColumn(
                                    "Vanna",
                                    format="%.6f",
                                    help=COLUMN_HELP["vanna"],
                                ),
                                "volga": st.column_config.NumberColumn(
                                    "Volga",
                                    format="$%.6f",
                                    help=COLUMN_HELP["volga"],
                                ),
                                "vomma": st.column_config.NumberColumn(
                                    "Vomma",
                                    format="$%.6f",
                                    help=COLUMN_HELP["vomma"],
                                ),
                                "charm": st.column_config.NumberColumn(
                                    "Charm/day",
                                    format="%.6f",
                                    help=COLUMN_HELP["charm"],
                                ),
                                "speed": st.column_config.NumberColumn(
                                    "Speed",
                                    format="%.6f",
                                    help=COLUMN_HELP["speed"],
                                ),
                                "color": st.column_config.NumberColumn(
                                    "Color/day",
                                    format="%.6f",
                                    help=COLUMN_HELP["color"],
                                ),
                            },
                        )
            st.caption(
                f"Showing {len(filtered):,} of {len(chain_df):,} valid contracts. "
                f"Source: {chain_meta.get('source', 'unknown')}; mode: {chain_meta.get('mode', 'unknown')}; "
                f"liquidity rejects: {chain_meta.get('liquidity_filtered_count', 0):,}; "
                f"price anatomy rows: {fmt_int(chain_meta.get('price_decomposition_contracts'))}; "
                f"pricing model: {chain_meta.get('pricing_model_label', 'n/a')}; "
                f"contract Greeks: {fmt_int(chain_meta.get('contract_greeks_count'))}; "
                f"second-order Greeks: {fmt_int(chain_meta.get('second_order_greeks_count'))}; "
                f"American model: {chain_meta.get('american_model', 'n/a')}; "
                f"early-exercise candidates: {fmt_int(chain_meta.get('early_exercise_candidates'))}."
            )
            scanner = surface_meta.get("rich_cheap_scanner") or {}
            scanner_display = pd.DataFrame(scanner.get("candidates") or [])
            if scanner.get("available") and not scanner_display.empty:
                st.markdown('<div class="section-header">Rich/Cheap Scanner</div>', unsafe_allow_html=True)
                scanner_cols = [
                    "classification",
                    "type",
                    "expiration",
                    "dte",
                    "strike",
                    "market_iv",
                    "fitted_iv",
                    "surface_residual",
                    "residual_z_score",
                    "liquidity_score",
                    "bid_ask_spread_pct",
                    "volume",
                    "open_interest",
                    "reason",
                ]
                scanner_cols = [col for col in scanner_cols if col in scanner_display.columns]
                scanner_table = coerce_table_numeric_columns(scanner_display[scanner_cols], SCANNER_NUMERIC_COLUMNS)
                scanner_display_table = format_scanner_table_for_display(scanner_display[scanner_cols])
                st.download_button(
                    "Export scanner CSV",
                    dataframe_to_csv_bytes(scanner_table),
                    file_name=f"{surface_symbol}_rich_cheap_scanner.csv",
                    mime="text/csv",
                )
                st.dataframe(
                    scanner_display_table,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "classification": st.column_config.TextColumn("Class"),
                        "type": st.column_config.TextColumn("Type", help=COLUMN_HELP["type"]),
                        "expiration": st.column_config.TextColumn("Expiration"),
                        "dte": st.column_config.TextColumn("DTE"),
                        "strike": st.column_config.TextColumn("Strike"),
                        "market_iv": st.column_config.TextColumn("Market IV"),
                        "fitted_iv": st.column_config.TextColumn("Fitted IV"),
                        "surface_residual": st.column_config.TextColumn("Residual"),
                        "residual_z_score": st.column_config.TextColumn("Z-score"),
                        "liquidity_score": st.column_config.TextColumn("Liquidity"),
                        "bid_ask_spread_pct": st.column_config.TextColumn("Spread"),
                        "volume": st.column_config.TextColumn("Volume"),
                        "open_interest": st.column_config.TextColumn("Open Interest"),
                        "reason": st.column_config.TextColumn("Reason"),
                    },
                )
                st.caption(
                    f"Scanner source: {scanner.get('source', 'current chain plus SVI fit')}; "
                    f"model {scanner.get('model', 'SVI')}; "
                    f"rich {fmt_int(scanner.get('rich_count'))}; cheap {fmt_int(scanner.get('cheap_count'))}."
                )
            else:
                st.markdown(
                    render_empty_state(
                        "Rich/cheap scanner unavailable",
                        scanner.get("reason") or "No current chain rows matched fitted-surface residuals.",
                        "Use a symbol with enough valid strikes for SVI calibration.",
                    ),
                    unsafe_allow_html=True,
                )
        elif show_chain:
            st.markdown(
                render_empty_state(
                    "Option chain unavailable",
                    "No usable yfinance contracts were returned after normalization and liquidity filters.",
                    "Use Refresh data, relax filters, or select another optionable symbol.",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                render_empty_state(
                    "Option chain panel idle",
                    "Enable Option chain in the configuration rail to render this table.",
                ),
                unsafe_allow_html=True,
            )

    with skew_tab:
        st.markdown('<div class="section-header">Smile And Term Structure</div>', unsafe_allow_html=True)
        smile_x, smile_vols, smile_days, smile_axis_title, smile_hover_label = extract_smile(
            strikes,
            expiries,
            vol_surface,
            current_data["price"],
            surface_x_axis,
        )
        expected_moves = surface_meta.get("expected_moves") or []
        expiry_events = surface_meta.get("expiry_events") or {}
        surface_change = surface_meta.get("surface_change") or {}
        col1, col2 = st.columns(2)
        with col1:
            fig_smile = go.Figure()
            fig_smile.add_trace(
                go.Scatter(
                    x=smile_x,
                    y=smile_vols,
                    mode="lines+markers",
                    name=f"{smile_days:.0f} DTE",
                    line=dict(color="#1f7a8c", width=3),
                    hovertemplate=f"{smile_hover_label}<br>IV: %{{y:.2%}}<extra></extra>",
                )
            )
            if surface_x_axis == "Strike":
                fig_smile.add_vline(x=current_data["price"], line_width=1, line_dash="dash", line_color="#667085")
            fig_smile.update_layout(
                title=f"{surface_symbol} Front Smile",
                xaxis_title=smile_axis_title,
                yaxis_title="Annualized IV",
            )
            st.plotly_chart(apply_chart_layout(fig_smile, 430), width="stretch")

        with col2:
            term = pd.DataFrame(stats.get("atm_term", []), columns=["DTE", "ATM IV"])
            fig_term = go.Figure()
            if not term.empty:
                fig_term.add_trace(
                    go.Scatter(
                        x=term["DTE"],
                        y=term["ATM IV"],
                        mode="lines+markers",
                        line=dict(color="#44546a", width=3),
                        hovertemplate="DTE: %{x:.0f}<br>ATM IV: %{y:.2%}<extra></extra>",
                    )
                )
                event_markers = _term_event_markers(term, expected_moves, expiry_events)
                if event_markers:
                    marker_frame = pd.DataFrame(event_markers)
                    fig_term.add_trace(
                        go.Scatter(
                            x=marker_frame["dte"],
                            y=marker_frame["atm_iv"],
                            mode="markers",
                            name="Events",
                            marker=dict(color="#c2410c", size=10, symbol="diamond"),
                            customdata=marker_frame[["label"]],
                            hovertemplate="DTE: %{x:.0f}<br>ATM IV: %{y:.2%}<br>%{customdata[0]}<extra></extra>",
                        )
                    )
            fig_term.update_layout(title=f"{surface_symbol} ATM Term Structure", xaxis_title="Days to expiry", yaxis_title="Annualized IV")
            st.plotly_chart(apply_chart_layout(fig_term, 430), width="stretch")

        st.caption(
            "Term structure: "
            f"{term_metrics.get('regime', 'unavailable')}; "
            f"front/back {fmt_pct(term_metrics.get('front_back_spread'))}; "
            f"slope per 30D {fmt_pct(term_metrics.get('slope_per_30d'))}; "
            f"curvature {term_metrics.get('curvature') if term_metrics.get('curvature') is not None else 'n/a'}."
        )
        event_rows = _event_rows(expiry_events)
        if event_rows:
            st.markdown('<div class="section-header">Event Expiry Annotations</div>', unsafe_allow_html=True)
            st.dataframe(
                pd.DataFrame(event_rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Expiry": st.column_config.TextColumn("Expiry"),
                    "Event Date": st.column_config.TextColumn("Event Date"),
                    "Type": st.column_config.TextColumn("Type"),
                    "Symbol": st.column_config.TextColumn("Symbol"),
                    "Description": st.column_config.TextColumn("Description"),
                    "Source": st.column_config.TextColumn("Source"),
                },
            )

        if surface_change.get("available"):
            st.markdown('<div class="section-header">Surface Change Vs Previous Snapshot</div>', unsafe_allow_html=True)
            atm_change = surface_change.get("atm_change") or {}
            vol_of_vol = surface_change.get("vol_of_vol") or {}
            st.caption(
                "Previous snapshot "
                f"{surface_change.get('previous_snapshot_timestamp', 'unknown')} "
                f"({surface_change.get('previous_snapshot_mode', 'unknown')}); "
                f"matched points {fmt_int(surface_change.get('matched_points'))}; "
                f"ATM dIV {fmt_pct(atm_change.get('iv_change'))}; "
                f"median absolute dIV {fmt_pct(surface_change.get('median_abs_iv_change'))}; "
                f"snapshot vol-of-vol {fmt_pct(vol_of_vol.get('snapshot_vol_of_vol'))}; "
                f"observations {fmt_int(vol_of_vol.get('observations'))}."
            )
            expiry_change_display = pd.DataFrame(surface_change.get("expiry_changes") or [])
            if not expiry_change_display.empty:
                fig_change = go.Figure(
                    data=[
                        go.Bar(
                            x=expiry_change_display["expiration"],
                            y=expiry_change_display["mean_iv_change"],
                            marker_color=np.where(
                                expiry_change_display["mean_iv_change"] >= 0,
                                "#087f5b",
                                "#b42318",
                            ),
                            hovertemplate="Expiry: %{x}<br>Mean dIV: %{y:.2%}<extra></extra>",
                        )
                    ]
                )
                fig_change.add_hline(y=0, line_width=1, line_color="#667085")
                fig_change.update_layout(
                    title=f"{surface_symbol} Mean IV Change By Expiry",
                    xaxis_title="Expiry",
                    yaxis_title="Current minus previous IV",
                )
                st.plotly_chart(apply_chart_layout(fig_change, 320), width="stretch")
                st.dataframe(
                    expiry_change_display,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "expiration": st.column_config.TextColumn("Expiry"),
                        "matched_points": st.column_config.NumberColumn("Matched", format="%d"),
                        "current_median_iv": st.column_config.NumberColumn("Current Median IV", format="%.2%"),
                        "previous_median_iv": st.column_config.NumberColumn("Previous Median IV", format="%.2%"),
                        "mean_iv_change": st.column_config.NumberColumn("Mean dIV", format="%.2%"),
                        "median_iv_change": st.column_config.NumberColumn("Median dIV", format="%.2%"),
                        "median_abs_iv_change": st.column_config.NumberColumn("Median Abs dIV", format="%.2%"),
                        "max_abs_iv_change": st.column_config.NumberColumn("Max Abs dIV", format="%.2%"),
                        "up_points": st.column_config.NumberColumn("Up", format="%d"),
                        "down_points": st.column_config.NumberColumn("Down", format="%d"),
                    },
                )
        else:
            st.markdown(
                render_empty_state(
                    "Surface change unavailable",
                    surface_change.get("reason")
                    or "No earlier persisted snapshot has matching expiry, strike, and type rows.",
                    "Refresh once to store a baseline, then refresh again to compare the current IV surface.",
                ),
                unsafe_allow_html=True,
            )

        delta_skew = surface_meta.get("delta_skew") or []
        if expected_moves:
            st.markdown('<div class="section-header">Expected Move By Expiry</div>', unsafe_allow_html=True)
            moves_display = pd.DataFrame(expected_moves)
            st.dataframe(
                moves_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "expiration": st.column_config.TextColumn("Expiry"),
                    "dte": st.column_config.NumberColumn("DTE", format="%.0f"),
                    "atm_strike": st.column_config.NumberColumn("ATM Strike", format="$%.2f"),
                    "atm_iv": st.column_config.NumberColumn("ATM IV", format="%.2%"),
                    "call_price": st.column_config.NumberColumn("Call", format="$%.2f"),
                    "put_price": st.column_config.NumberColumn("Put", format="$%.2f"),
                    "straddle_move": st.column_config.NumberColumn("Straddle Move", format="$%.2f"),
                    "iv_move": st.column_config.NumberColumn("IV Move", format="$%.2f"),
                    "expected_move": st.column_config.NumberColumn("Expected Move", format="$%.2f"),
                    "expected_move_pct": st.column_config.NumberColumn("Move %", format="%.2%"),
                    "lower_bound": st.column_config.NumberColumn("Lower", format="$%.2f"),
                    "upper_bound": st.column_config.NumberColumn("Upper", format="$%.2f"),
                    "method": st.column_config.TextColumn("Method"),
                    "confidence": st.column_config.TextColumn("Confidence"),
                },
            )
        else:
            st.markdown(
                render_empty_state(
                    "Expected move unavailable",
                    "The current chain does not have enough ATM price or IV data to estimate expiry moves.",
                    "Refresh data, relax filters, or select a more liquid options symbol.",
                ),
                unsafe_allow_html=True,
            )

        if delta_skew:
            skew_display = pd.DataFrame(delta_skew)
            st.dataframe(
                skew_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "expiration": st.column_config.TextColumn("Expiry"),
                    "dte": st.column_config.NumberColumn("DTE", format="%.0f"),
                    "atm_iv": st.column_config.NumberColumn("ATM IV", format="%.2%"),
                    "10d_put_iv": st.column_config.NumberColumn("10d Put IV", format="%.2%"),
                    "25d_put_iv": st.column_config.NumberColumn("25d Put IV", format="%.2%"),
                    "25d_call_iv": st.column_config.NumberColumn("25d Call IV", format="%.2%"),
                    "10d_call_iv": st.column_config.NumberColumn("10d Call IV", format="%.2%"),
                    "risk_reversal_25d": st.column_config.NumberColumn("25d RR", format="%.2%"),
                    "butterfly_25d": st.column_config.NumberColumn("25d Fly", format="%.2%"),
                },
            )
        else:
            st.markdown(
                render_empty_state(
                    "Delta skew unavailable",
                    "The current chain does not have enough valid call and put IVs to compute 10/25-delta skew.",
                    "Refresh data, relax filters, or select a more liquid options symbol.",
                ),
                unsafe_allow_html=True,
            )

        svi_smiles = surface_meta.get("svi_smiles") or []
        fit_diagnostics = surface_meta.get("fit_diagnostics") or {}
        fit_mode_comparison = surface_meta.get("fit_mode_comparison") or []
        if fit_mode_comparison:
            st.markdown('<div class="section-header">Surface Fit Mode Comparison</div>', unsafe_allow_html=True)
            fit_comparison_rows = fit_comparison_display_rows(surface_meta)
            fit_comparison_frame = pd.DataFrame(fit_comparison_rows)
            st.dataframe(
                fit_comparison_frame,
                width="stretch",
                hide_index=True,
                column_config={
                    "fit_mode": st.column_config.TextColumn("Fit Mode"),
                    "status": st.column_config.TextColumn("Status"),
                    "eligible_rows": st.column_config.NumberColumn("Eligible Rows", format="%d"),
                    "excluded_rows": st.column_config.NumberColumn("Excluded Rows", format="%d"),
                    "weighted_rmse": st.column_config.NumberColumn("Weighted RMSE", format="%.2%"),
                    "unweighted_rmse": st.column_config.NumberColumn("Unweighted RMSE", format="%.2%"),
                    "no_arb_violations": st.column_config.NumberColumn("No-Arb Violations", format="%d"),
                    "prior_weight": st.column_config.NumberColumn("Prior Weight", format="%.2%"),
                    "ml_uncertainty": st.column_config.NumberColumn("ML Uncertainty", format="%.2%"),
                    "timestamp": st.column_config.TextColumn("Timestamp"),
                    "provenance": st.column_config.TextColumn("Provenance"),
                },
            )
            st.download_button(
                "Export fit comparison CSV",
                dataframe_to_csv_bytes(fit_comparison_frame),
                file_name=f"{surface_symbol}_fit_comparison.csv",
                mime="text/csv",
                key="fit_comparison_export_csv",
            )
        if svi_smiles:
            st.markdown('<div class="section-header">SVI Fit Diagnostics</div>', unsafe_allow_html=True)
            svi_display = pd.DataFrame(svi_smiles).drop(columns=["residuals", "residual_diagnostics"], errors="ignore")
            st.dataframe(
                svi_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "expiration": st.column_config.TextColumn("Expiry"),
                    "dte": st.column_config.NumberColumn("DTE", format="%.0f"),
                    "points": st.column_config.NumberColumn("Points", format="%d"),
                    "a": st.column_config.NumberColumn("a", format="%.5f"),
                    "b": st.column_config.NumberColumn("b", format="%.5f"),
                    "rho": st.column_config.NumberColumn("rho", format="%.4f"),
                    "m": st.column_config.NumberColumn("m", format="%.4f"),
                    "sigma": st.column_config.NumberColumn("sigma", format="%.4f"),
                    "rmse": st.column_config.NumberColumn("RMSE", format="%.2%"),
                    "weighted_rmse": st.column_config.NumberColumn("Weighted RMSE", format="%.2%"),
                    "mae": st.column_config.NumberColumn("MAE", format="%.2%"),
                    "max_error": st.column_config.NumberColumn("Max Error", format="%.2%"),
                    "weight_mode": st.column_config.TextColumn("Weight Mode"),
                    "loss_mode": st.column_config.TextColumn("Loss"),
                },
            )
            residual_diagnostics = fit_diagnostics.get("residual_diagnostics") or {}
            top_residuals = pd.DataFrame(residual_diagnostics.get("top_residuals") or [])
            if not top_residuals.empty:
                st.dataframe(
                    top_residuals,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "expiration": st.column_config.TextColumn("Expiry"),
                        "dte": st.column_config.NumberColumn("DTE", format="%.0f"),
                        "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                        "log_moneyness": st.column_config.NumberColumn("Log-moneyness", format="%.3f"),
                        "observed_iv": st.column_config.NumberColumn("Observed IV", format="%.2%"),
                        "fitted_iv": st.column_config.NumberColumn("Fitted IV", format="%.2%"),
                        "residual": st.column_config.NumberColumn("Residual", format="%.2%"),
                        "abs_residual": st.column_config.NumberColumn("Abs Residual", format="%.2%"),
                        "clipped_residual": st.column_config.NumberColumn("Clipped Residual", format="%.2%"),
                        "fit_weight": st.column_config.NumberColumn("Fit Weight", format="%.3f"),
                        "clipped": st.column_config.CheckboxColumn("Clipped"),
                        "downweighted": st.column_config.CheckboxColumn("Downweighted"),
                    },
                )
            front_residuals = pd.DataFrame(svi_smiles[0].get("residuals") or [])
            if not front_residuals.empty:
                fig_residual = go.Figure()
                fig_residual.add_trace(
                    go.Bar(
                        x=front_residuals["log_moneyness"],
                        y=front_residuals["residual"],
                        marker_color="#9b5de5",
                        name="Residual",
                        hovertemplate=(
                            "Log-moneyness: %{x:.3f}<br>"
                            "Residual: %{y:.2%}<br>"
                            "Observed IV: %{customdata[0]:.2%}<br>"
                            "Fitted IV: %{customdata[1]:.2%}<extra></extra>"
                        ),
                        customdata=front_residuals[["observed_iv", "fitted_iv"]],
                    )
                )
                fig_residual.add_hline(y=0, line_width=1, line_color="#667085")
                fig_residual.update_layout(
                    title=f"{surface_symbol} Front SVI Residuals",
                    xaxis_title="Log-moneyness",
                    yaxis_title="Fitted minus raw IV",
                )
                st.plotly_chart(apply_chart_layout(fig_residual, 360), width="stretch")
            st.caption(
                "SVI fit quality: "
                f"expiries {fmt_int(fit_diagnostics.get('fitted_expiries'))}; "
                f"points {fmt_int(fit_diagnostics.get('points'))}; "
                f"RMSE {fmt_pct(fit_diagnostics.get('rmse'))}; "
                f"weighted RMSE {fmt_pct(fit_diagnostics.get('weighted_rmse'))}; "
                f"clipped {fmt_int(residual_diagnostics.get('clipped_count'))}; "
                f"downweighted {fmt_int(residual_diagnostics.get('downweighted_count'))}; "
                f"MAE {fmt_pct(fit_diagnostics.get('mae'))}; "
                f"max error {fmt_pct(fit_diagnostics.get('max_error'))}."
            )
        else:
            st.markdown(
                render_empty_state(
                    "SVI fit unavailable",
                    "No expiry has enough valid IV points for deterministic SVI calibration.",
                    "Refresh data, relax filters, or select a more liquid options symbol.",
                ),
                unsafe_allow_html=True,
            )

        ssvi_surface = surface_meta.get("ssvi_surface") or {}
        global_fit = surface_meta.get("global_fit_diagnostics") or {}
        if ssvi_surface.get("status") == "fitted":
            st.markdown('<div class="section-header">Global SSVI Surface Fit</div>', unsafe_allow_html=True)
            ssvi_rows = pd.DataFrame(ssvi_surface.get("atm_total_variance") or [])
            if not ssvi_rows.empty:
                st.dataframe(
                    ssvi_rows,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "expiration": st.column_config.TextColumn("Expiry"),
                        "dte": st.column_config.NumberColumn("DTE", format="%.0f"),
                        "theta": st.column_config.NumberColumn("ATM Total Variance", format="%.5f"),
                        "raw_theta": st.column_config.NumberColumn("Raw ATM Total Variance", format="%.5f"),
                        "points": st.column_config.NumberColumn("Points", format="%d"),
                    },
                )
            constraints = ssvi_surface.get("constraints") or {}
            ssvi_residual_diagnostics = ssvi_surface.get("residual_diagnostics") or {}
            ssvi_top_residuals = pd.DataFrame(ssvi_residual_diagnostics.get("top_residuals") or [])
            if not ssvi_top_residuals.empty:
                st.dataframe(
                    ssvi_top_residuals,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "expiration": st.column_config.TextColumn("Expiry"),
                        "dte": st.column_config.NumberColumn("DTE", format="%.0f"),
                        "strike": st.column_config.NumberColumn("Strike", format="%.2f"),
                        "log_moneyness": st.column_config.NumberColumn("Log-moneyness", format="%.3f"),
                        "observed_iv": st.column_config.NumberColumn("Observed IV", format="%.2%"),
                        "fitted_iv": st.column_config.NumberColumn("Fitted IV", format="%.2%"),
                        "residual": st.column_config.NumberColumn("Residual", format="%.2%"),
                        "fit_weight": st.column_config.NumberColumn("Fit Weight", format="%.3f"),
                        "clipped": st.column_config.CheckboxColumn("Clipped"),
                        "downweighted": st.column_config.CheckboxColumn("Downweighted"),
                    },
                )
            st.caption(
                "SSVI global fit: "
                f"rho {ssvi_surface.get('rho'):.4f}; "
                f"eta {ssvi_surface.get('eta'):.4f}; "
                f"gamma {ssvi_surface.get('gamma'):.4f}; "
                f"expiries {fmt_int(global_fit.get('fitted_expiries'))}; "
                f"points {fmt_int(global_fit.get('points'))}; "
                f"RMSE {fmt_pct(global_fit.get('rmse'))}; "
                f"weighted RMSE {fmt_pct(global_fit.get('weighted_rmse'))}; "
                f"clipped {fmt_int(ssvi_residual_diagnostics.get('clipped_count'))}; "
                f"downweighted {fmt_int(ssvi_residual_diagnostics.get('downweighted_count'))}; "
                f"constraints passed {constraints.get('passed')}."
            )
        else:
            st.markdown(
                render_empty_state(
                    "SSVI global fit unavailable",
                    ssvi_surface.get("reason") or "The current chain does not have enough valid expiries for global SSVI calibration.",
                    "Refresh data, relax filters, or use a symbol with at least two liquid expiries.",
                ),
                unsafe_allow_html=True,
            )

        hist_metrics = load_with_status(
            st,
            LoadingState(
                title=f"{surface_symbol} historical volatility",
                detail="Fetching yfinance historical closes for realized-volatility comparison.",
                stage="history",
                rows=4,
            ),
            lambda: get_historical_metrics_cached(surface_symbol),
        )
        if hist_metrics.get("available"):
            r20 = hist_metrics.get("realized_20d_latest")
            r60 = hist_metrics.get("realized_60d_latest")
            st.caption(
                f"Realized volatility from {hist_metrics.get('source')}: "
                f"20D {fmt_pct(r20)}, 60D {fmt_pct(r60)}."
            )
            realized_latest = hist_metrics.get("realized_estimator_latest") or {}
            if realized_latest:
                rows = []
                for label, prefix in (
                    ("Close-to-close", "close_to_close"),
                    ("Parkinson", "parkinson"),
                    ("Garman-Klass", "garman_klass"),
                    ("Rogers-Satchell", "rogers_satchell"),
                    ("Yang-Zhang", "yang_zhang"),
                ):
                    rows.append(
                        {
                            "Estimator": label,
                            "20D": realized_latest.get(f"{prefix}_20d"),
                            "60D": realized_latest.get(f"{prefix}_60d"),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Estimator": st.column_config.TextColumn("Estimator"),
                        "20D": st.column_config.NumberColumn("20D", format="%.2%"),
                        "60D": st.column_config.NumberColumn("60D", format="%.2%"),
                    },
                )
        else:
            st.markdown(
                render_empty_state(
                    "Historical volatility unavailable",
                    f"Realized-vol fetch did not return enough usable closes: {hist_metrics.get('reason', 'unknown reason')}.",
                    "Refresh data or switch to a symbol with liquid yfinance history.",
                ),
                unsafe_allow_html=True,
            )

    with term_tab:
        st.markdown('<div class="section-header">Term Structure Panel</div>', unsafe_allow_html=True)
        term = pd.DataFrame(stats.get("atm_term", []), columns=["DTE", "ATM IV"])
        term_fig = go.Figure()
        if not term.empty:
            term_fig.add_trace(
                go.Scatter(
                    x=term["DTE"],
                    y=term["ATM IV"],
                    mode="lines+markers",
                    name="ATM IV",
                    line=dict(color="#176B87", width=3),
                    hovertemplate="DTE: %{x:.0f}<br>ATM IV: %{y:.2%}<extra></extra>",
                )
            )
            if hist_metrics.get("available"):
                r20 = hist_metrics.get("realized_20d_latest")
                r60 = hist_metrics.get("realized_60d_latest")
                if r20 is not None:
                    term_fig.add_hline(
                        y=float(r20),
                        line_width=1,
                        line_dash="dash",
                        line_color="#b42318",
                        annotation_text="20D realized",
                    )
                if r60 is not None:
                    term_fig.add_hline(
                        y=float(r60),
                        line_width=1,
                        line_dash="dot",
                        line_color="#7a5af8",
                        annotation_text="60D realized",
                    )
            event_markers = _term_event_markers(term, expected_moves, expiry_events)
            if event_markers:
                marker_frame = pd.DataFrame(event_markers)
                term_fig.add_trace(
                    go.Scatter(
                        x=marker_frame["dte"],
                        y=marker_frame["atm_iv"],
                        mode="markers",
                        name="Events",
                        marker=dict(color="#c2410c", size=11, symbol="diamond"),
                        customdata=marker_frame[["label"]],
                        hovertemplate="DTE: %{x:.0f}<br>ATM IV: %{y:.2%}<br>%{customdata[0]}<extra></extra>",
                    )
                )
            term_fig.update_layout(
                title=f"{surface_symbol} ATM IV Term Structure",
                xaxis_title="Days to expiry",
                yaxis_title="Annualized volatility",
            )
            st.plotly_chart(apply_chart_layout(term_fig, 460), width="stretch")
            st.caption(
                f"Front/back spread {fmt_pct(term_metrics.get('front_back_spread'))}; "
                f"slope per 30D {fmt_pct(term_metrics.get('slope_per_30d'))}; "
                f"regime {term_metrics.get('regime', 'unavailable')}; "
                f"surface source {surface_meta.get('surface_source', 'unknown')}."
            )
        else:
            st.markdown(
                render_empty_state(
                    "Term structure unavailable",
                    "No ATM IV points were available across expiries.",
                    "Refresh data or choose a symbol with multiple valid expirations.",
                ),
                unsafe_allow_html=True,
            )
        event_rows = _event_rows(expiry_events)
        if event_rows:
            st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)

    with quality_tab:
        st.markdown('<div class="section-header">Data Quality Panel</div>', unsafe_allow_html=True)
        quality_cols = st.columns(5)
        quality_metrics = [
            ("Source", surface_meta.get("surface_source") or current_data.get("price_source") or "unknown", surface_mode),
            ("Cache Age", f"{fmt_int(surface_meta.get('cache_age_seconds'))}s", "chain"),
            ("Rejected Rows", fmt_int(surface_meta.get("rejected_rows")), "normalization"),
            ("No-Arb", fmt_int(surface_meta.get("no_arbitrage_violation_count")), "violations"),
            ("Fit RMSE", fmt_pct((surface_meta.get("fit_diagnostics") or {}).get("rmse")), "SVI"),
        ]
        for col, (label, value, delta) in zip(quality_cols, quality_metrics):
            with col:
                st.metric(label, value, delta=delta)
        quality_alert = quality_drop_alert_summary(surface_meta)
        if quality_alert["level"] == "warning":
            st.warning(quality_alert["message"])
        elif quality_alert["level"] == "success":
            st.success(quality_alert["message"])
        else:
            st.info(quality_alert["message"])
        actionability = data_quality_actionability(surface_meta)
        action_cols = st.columns(2)
        with action_cols[0]:
            penalty_frame = pd.DataFrame(actionability["top_penalty_reasons"])
            if not penalty_frame.empty:
                st.markdown("#### Top Quality Drivers")
                st.dataframe(
                    penalty_frame,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "reason": st.column_config.TextColumn("Reason"),
                        "count": st.column_config.NumberColumn("Count", format="%d"),
                    },
                )
        with action_cols[1]:
            no_arb_summary = actionability["no_arbitrage"]
            st.markdown("#### No-Arbitrage Summary")
            st.json(no_arb_summary)
        worst_expiries = pd.DataFrame(actionability["worst_expiries"])
        if not worst_expiries.empty:
            st.markdown("#### Worst Expiries")
            st.dataframe(
                worst_expiries,
                width="stretch",
                hide_index=True,
                column_config={
                    "score": st.column_config.NumberColumn("Score", format="%.1f"),
                    "surface_quotes": st.column_config.NumberColumn("Surface Quotes", format="%d"),
                    "rejected_quotes": st.column_config.NumberColumn("Rejected Quotes", format="%d"),
                },
            )
        worst_residuals = pd.DataFrame(actionability["worst_residual_contracts"])
        if not worst_residuals.empty:
            st.markdown("#### Worst Residual Contracts")
            st.dataframe(
                worst_residuals,
                width="stretch",
                hide_index=True,
                column_config={
                    "contract": st.column_config.TextColumn("Contract"),
                    "type": st.column_config.TextColumn("Type"),
                    "expiration": st.column_config.TextColumn("Expiry"),
                    "dte": st.column_config.NumberColumn("DTE", format="%.0f"),
                    "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                    "observed_iv": st.column_config.NumberColumn("Observed IV", format="%.2%"),
                    "fitted_iv": st.column_config.NumberColumn("Fitted IV", format="%.2%"),
                    "residual": st.column_config.NumberColumn("Residual", format="%.2%"),
                    "abs_residual": st.column_config.NumberColumn("Abs Residual", format="%.2%"),
                    "fit_weight": st.column_config.NumberColumn("Fit Weight", format="%.3f"),
                    "source": st.column_config.TextColumn("Source"),
                },
            )
        st.caption(f"Suggested fit preset: {actionability['suggested_preset']}.")
        expiry_quality = surface_meta.get("expiry_quality") or {}
        if expiry_quality:
            quality_rows = []
            for expiry, payload in sorted(expiry_quality.items()):
                buckets = payload.get("reason_buckets") or {}
                quality_rows.append(
                    {
                        "Expiry": expiry,
                        "Score": payload.get("score"),
                        "Valid Quotes": payload.get("valid_quotes"),
                        "Rejected Quotes": payload.get("rejected_quotes"),
                        "Surface Quotes": payload.get("surface_quotes"),
                        "Reason Buckets": ", ".join(
                            f"{reason}: {count}" for reason, count in sorted(buckets.items()) if count
                        )
                        or "none",
                    }
                )
            st.dataframe(
                pd.DataFrame(quality_rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Score": st.column_config.NumberColumn("Score", format="%.1f"),
                    "Valid Quotes": st.column_config.NumberColumn("Valid Quotes", format="%d"),
                    "Rejected Quotes": st.column_config.NumberColumn("Rejected Quotes", format="%d"),
                    "Surface Quotes": st.column_config.NumberColumn("Surface Quotes", format="%d"),
                },
            )
        else:
            st.markdown(
                render_empty_state(
                    "Expiry quality unavailable",
                    "The current surface metadata has no per-expiry quality buckets.",
                    "Refresh data to rebuild quote-quality diagnostics.",
                ),
                unsafe_allow_html=True,
            )
        quality_label = f"quality score {quality_score:.1f}/100" if quality_score is not None else "quality score unavailable"
        st.caption(
            f"Timestamp {surface_meta.get('timestamp') or surface_meta.get('spot_timestamp') or current_data.get('timestamp')}; "
            f"{quality_label}; "
            f"reason buckets {reason_buckets or {}}; "
            f"fallback reason {surface_meta.get('fallback_reason') or 'none'}."
        )
        surface_quality = surface_meta.get("surface_quality") or {}
        if surface_quality:
            st.json(surface_quality)

    with local_vol_tab:
        st.markdown('<div class="section-header">Dupire Local Volatility</div>', unsafe_allow_html=True)
        local_vol = surface_meta.get("local_volatility") or {}
        if local_vol.get("enabled"):
            local_grid = np.asarray(local_vol.get("grid"), dtype=float)
            fig_local = go.Figure(
                data=[
                    go.Heatmap(
                        z=local_grid,
                        x=np.asarray(strikes)[0, :] if np.asarray(strikes).ndim == 2 else strikes,
                        y=np.asarray(expiries)[:, 0] if np.asarray(expiries).ndim == 2 else expiries,
                        colorscale="Cividis",
                        colorbar=dict(title="Local vol"),
                        hovertemplate="Strike: %{x:.2f}<br>DTE: %{y:.0f}<br>Local vol: %{z:.2%}<extra></extra>",
                    )
                ]
            )
            fig_local.update_layout(
                title=f"{surface_symbol} Dupire Local Vol Approximation",
                xaxis_title="Strike",
                yaxis_title="Days to expiry",
            )
            st.plotly_chart(apply_chart_layout(fig_local, 460), width="stretch")
            st.caption(
                "Local vol diagnostics: "
                f"min {fmt_pct(local_vol.get('min_local_vol'))}; "
                f"max {fmt_pct(local_vol.get('max_local_vol'))}; "
                f"invalid points {fmt_int(local_vol.get('invalid_points'))}. "
                + " ".join(str(item) for item in local_vol.get("warnings", []))
            )
        else:
            st.markdown(
                render_empty_state(
                    "Local volatility disabled",
                    local_vol.get("reason") or "Dupire local vol requires a smoothed high-quality surface.",
                    "Improve quote quality, surface density, and smoothing diagnostics before relying on this approximation.",
                ),
                unsafe_allow_html=True,
            )

    with relative_tab:
        st.markdown('<div class="section-header">Scanner Panel</div>', unsafe_allow_html=True)
        scanner = surface_meta.get("rich_cheap_scanner") or {}
        scanner_display = pd.DataFrame(scanner.get("candidates") or [])
        if scanner.get("available") and not scanner_display.empty:
            scanner_cols = [
                "classification",
                "type",
                "expiration",
                "dte",
                "strike",
                "market_iv",
                "fitted_iv",
                "surface_residual",
                "residual_z_score",
                "liquidity_score",
                "bid_ask_spread_pct",
                "volume",
                "open_interest",
                "reason",
            ]
            scanner_cols = [col for col in scanner_cols if col in scanner_display.columns]
            scanner_table = coerce_table_numeric_columns(scanner_display[scanner_cols], SCANNER_NUMERIC_COLUMNS)
            scanner_display_table = format_scanner_table_for_display(scanner_display[scanner_cols])
            st.download_button(
                "Export scanner panel CSV",
                dataframe_to_csv_bytes(scanner_table),
                file_name=f"{surface_symbol}_scanner_panel.csv",
                mime="text/csv",
                key="scanner_panel_export_csv",
            )
            st.dataframe(
                scanner_display_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "classification": st.column_config.TextColumn("Class"),
                    "dte": st.column_config.TextColumn("DTE"),
                    "strike": st.column_config.TextColumn("Strike"),
                    "market_iv": st.column_config.TextColumn("Market IV"),
                    "fitted_iv": st.column_config.TextColumn("Fitted IV"),
                    "surface_residual": st.column_config.TextColumn("Residual"),
                    "residual_z_score": st.column_config.TextColumn("Z-score"),
                    "liquidity_score": st.column_config.TextColumn("Liquidity"),
                    "bid_ask_spread_pct": st.column_config.TextColumn("Spread"),
                    "volume": st.column_config.TextColumn("Volume"),
                    "open_interest": st.column_config.TextColumn("Open Interest"),
                    "reason": st.column_config.TextColumn("Reason"),
                },
            )
            st.caption(
                f"Residual scanner source: {scanner.get('source', 'current chain plus SVI fit')}; "
                f"model {scanner.get('model', 'SVI')}; "
                f"rich {fmt_int(scanner.get('rich_count'))}; cheap {fmt_int(scanner.get('cheap_count'))}; "
                f"surface mode {surface_meta.get('surface_mode', 'unknown')}."
            )
        else:
            st.markdown(
                render_empty_state(
                    "Residual scanner unavailable",
                    scanner.get("reason") or "No current chain rows matched fitted-surface residuals.",
                    "Use a symbol with enough valid strikes for SVI calibration.",
                ),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Relative Value Dashboard</div>', unsafe_allow_html=True)
        peer_symbols = [symbol for symbol in selected_symbols if symbol != surface_symbol]
        if peer_symbols:
            peer_symbol = st.selectbox("Peer underlying", peer_symbols, index=0, help="Second symbol for pair overlays.")
            rv = load_with_status(
                st,
                LoadingState(
                    title=f"{surface_symbol} / {peer_symbol} relative value",
                    detail="Building pair metrics from option-chain, surface, and realized-volatility profiles.",
                    stage="relative value",
                    rows=4,
                ),
                lambda: get_relative_value_cached(surface_symbol, peer_symbol, data_key),
            )
            if rv.get("available"):
                overlay = pd.DataFrame(rv.get("normalized_overlays") or [])
                profiles = pd.DataFrame(rv.get("profiles") or [])
                if not overlay.empty:
                    fig_overlay = go.Figure()
                    fig_overlay.add_trace(
                        go.Bar(
                            x=overlay["metric"],
                            y=overlay["left_normalized"],
                            name=surface_symbol,
                            marker_color="#176B87",
                        )
                    )
                    fig_overlay.add_trace(
                        go.Bar(
                            x=overlay["metric"],
                            y=overlay["right_normalized"],
                            name=peer_symbol,
                            marker_color="#F59E0B",
                        )
                    )
                    fig_overlay.update_layout(
                        title=f"{surface_symbol} vs {peer_symbol} Normalized Vol Overlay",
                        yaxis_title="Normalized value",
                        barmode="group",
                    )
                    st.plotly_chart(apply_chart_layout(fig_overlay, 420), width="stretch")
                if not profiles.empty:
                    st.dataframe(
                        profiles,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "symbol": st.column_config.TextColumn("Symbol"),
                            "atm_iv": st.column_config.NumberColumn("ATM IV", format="%.2%"),
                            "iv_rank": st.column_config.NumberColumn("IV Rank", format="%.2%"),
                            "iv_percentile": st.column_config.NumberColumn("IV Percentile", format="%.2%"),
                            "skew_25d": st.column_config.NumberColumn("25D Skew", format="%.2%"),
                            "term_slope": st.column_config.NumberColumn("Term Slope", format="%.2%"),
                            "realized_20d": st.column_config.NumberColumn("20D Realized", format="%.2%"),
                            "iv_realized_spread": st.column_config.NumberColumn("IV - Realized", format="%.2%"),
                            "mode": st.column_config.TextColumn("Mode"),
                            "source": st.column_config.TextColumn("Source"),
                        },
                    )
                    st.caption(
                        f"Pair source: {rv.get('source', 'symbol profiles')}; "
                        f"ATM spread {fmt_pct((rv.get('spreads') or {}).get('atm_iv_spread'))}; "
                        f"skew spread {fmt_pct((rv.get('spreads') or {}).get('skew_spread'))}; "
                        f"realized spread {fmt_pct((rv.get('spreads') or {}).get('realized_spread'))}."
                    )
            else:
                st.markdown(
                    render_empty_state(
                        "Relative value unavailable",
                        rv.get("reason") or "Pair profiles did not include enough volatility metrics.",
                        "Select two optionable symbols with surface and realized-volatility data.",
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                render_empty_state(
                    "Peer selection unavailable",
                    "Relative value needs at least two selected symbols.",
                    "Add another symbol to the universe.",
                ),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Cross-Sectional Vol Map</div>', unsafe_allow_html=True)
        xvol = load_with_status(
            st,
            LoadingState(
                title="Cross-sectional volatility map",
                detail="Ranking selected symbols by IV rank, percentile, skew, term slope, and IV-realized spread.",
                stage="cross-section",
                rows=max(3, min(len(selected_symbols), 8)),
            ),
            lambda: get_cross_sectional_vol_map_cached(tuple(sorted(selected_symbols)), data_key),
        )
        xvol_rows = pd.DataFrame(xvol.get("opportunities") or [])
        if xvol.get("available") and not xvol_rows.empty:
            min_score = st.slider(
                "Min opportunity score",
                0.0,
                max(1.0, float(xvol_rows["opportunity_score"].max())),
                0.0,
                0.05,
                help="Minimum cross-sectional score shown in the map.",
            )
            xvol_display = xvol_rows[xvol_rows["opportunity_score"] >= min_score].copy()
            fig_xvol = go.Figure(
                data=go.Scatter(
                    x=xvol_display["iv_realized_spread"],
                    y=xvol_display["iv_rank"],
                    mode="markers+text",
                    text=xvol_display["symbol"],
                    textposition="top center",
                    marker=dict(
                        size=np.clip(xvol_display["opportunity_score"].astype(float) * 18 + 8, 8, 34),
                        color=xvol_display["term_slope"],
                        colorscale="Cividis",
                        colorbar=dict(title="Term slope"),
                    ),
                    customdata=xvol_display[["iv_percentile", "skew_25d", "opportunity_score"]],
                    hovertemplate=(
                        "%{text}<br>IV-realized: %{x:.2%}<br>IV rank: %{y:.2%}<br>"
                        "IV percentile: %{customdata[0]:.2%}<br>25D skew: %{customdata[1]:.2%}<br>"
                        "Score: %{customdata[2]:.2f}<extra></extra>"
                    ),
                )
            )
            fig_xvol.update_layout(
                title="Universe Vol Opportunity Map",
                xaxis_title="IV - 20D realized",
                yaxis_title="IV rank",
            )
            st.plotly_chart(apply_chart_layout(fig_xvol, 440), width="stretch")
            st.download_button(
                "Export vol map CSV",
                dataframe_to_csv_bytes(xvol_display),
                file_name=f"{surface_symbol}_cross_sectional_vol_map.csv",
                mime="text/csv",
            )
            st.dataframe(
                xvol_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "symbol": st.column_config.TextColumn("Symbol"),
                    "atm_iv": st.column_config.NumberColumn("ATM IV", format="%.2%"),
                    "iv_rank": st.column_config.NumberColumn("IV Rank", format="%.2%"),
                    "iv_percentile": st.column_config.NumberColumn("IV Percentile", format="%.2%"),
                    "skew_25d": st.column_config.NumberColumn("25D Skew", format="%.2%"),
                    "term_slope": st.column_config.NumberColumn("Term Slope", format="%.2%"),
                    "iv_realized_spread": st.column_config.NumberColumn("IV - Realized", format="%.2%"),
                    "opportunity_score": st.column_config.NumberColumn("Score", format="%.2f"),
                    "mode": st.column_config.TextColumn("Mode"),
                    "source": st.column_config.TextColumn("Source"),
                },
            )
            st.caption(
                f"Map source: {xvol.get('source', 'symbol profiles')}; "
                f"symbols ranked {fmt_int(xvol.get('symbol_count'))}; "
                f"metrics: {', '.join(xvol.get('metrics') or [])}."
            )
        else:
            st.markdown(
                render_empty_state(
                    "Cross-sectional map unavailable",
                    xvol.get("reason") or "No selected symbol has enough volatility metrics.",
                    "Refresh data or select optionable symbols with history.",
                ),
                unsafe_allow_html=True,
            )

    with strategy_tab:
        st.markdown('<div class="section-header">Earnings Vol Event Engine</div>', unsafe_allow_html=True)
        event_payload = load_with_status(
            st,
            LoadingState(
                title=f"{surface_symbol} earnings event card",
                detail="Matching earnings calendar rows to expected-move and ATM term-structure inputs.",
                stage="earnings event",
                rows=3,
            ),
            lambda: get_earnings_event_cached(surface_symbol, data_key),
        )
        if event_payload.get("available"):
            card = event_payload.get("event_card") or {}
            event_cols = st.columns(5)
            event_metrics = [
                ("Event", card.get("event_date") or "n/a", card.get("description")),
                ("Implied Move", fmt_money(card.get("implied_move")), fmt_pct(card.get("implied_move_pct"))),
                ("Hist Move", fmt_pct(card.get("historical_avg_abs_move_pct")), "avg abs"),
                ("Crush", fmt_pct(card.get("post_event_crush")), "ATM term"),
                ("Expiry", card.get("expiration") or "n/a", f"DTE {fmt_int(card.get('dte'))}"),
            ]
            for col, (label, value, delta) in zip(event_cols, event_metrics):
                with col:
                    st.metric(label, value, delta=delta if delta else None)
            st.caption(
                f"Event source: {event_payload.get('source', 'event calendar plus option chain')}; "
                f"method {card.get('method', 'n/a')}; "
                f"historical observations {fmt_int(event_payload.get('historical_observations'))}."
            )
        else:
            st.markdown(
                render_empty_state(
                    "Earnings card unavailable",
                    event_payload.get("reason") or "No upcoming earnings event matched the current chain.",
                    "Add a local earnings row in the event calendar or select a symbol with an upcoming event.",
                ),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Strategy Builder</div>', unsafe_allow_html=True)
        strategy_type = st.selectbox(
            "Strategy",
            ["straddle", "strangle", "vertical", "calendar", "diagonal", "butterfly", "condor", "risk reversal"],
            index=0,
            help="Template strategy to build from the current option chain.",
        )
        strategy = load_with_status(
            st,
            LoadingState(
                title=f"{surface_symbol} {strategy_type} strategy",
                detail="Selecting template legs and repricing each leg with fitted surface IV.",
                stage="strategy",
                rows=4,
            ),
            lambda: get_strategy_analytics_cached(surface_symbol, strategy_type, data_key),
        )
        if strategy.get("available"):
            strategy_cols = st.columns(5)
            strategy_metrics = [
                ("Net Debit", fmt_money(strategy.get("net_debit")), fmt_money(strategy.get("net_debit_100x"))),
                ("Delta", _fmt_number((strategy.get("greeks") or {}).get("delta")), "net"),
                ("Gamma", _fmt_number((strategy.get("greeks") or {}).get("gamma")), "net"),
                ("Theta/day", fmt_money((strategy.get("greeks") or {}).get("theta")), "net"),
                ("Vega/1%", fmt_money((strategy.get("greeks") or {}).get("vega")), "net"),
            ]
            for col, (label, value, delta) in zip(strategy_cols, strategy_metrics):
                with col:
                    st.metric(label, value, delta=delta if delta else None)
            payoff = pd.DataFrame(strategy.get("payoff_points") or [])
            if not payoff.empty:
                fig_payoff = go.Figure()
                fig_payoff.add_trace(
                    go.Scatter(
                        x=payoff["spot"],
                        y=payoff["pnl"],
                        mode="lines",
                        name="P&L",
                        line=dict(color="#176B87", width=3),
                    )
                )
                fig_payoff.add_hline(y=0, line_width=1, line_color="#667085")
                fig_payoff.update_layout(
                    title=f"{surface_symbol} {strategy_type.title()} Payoff",
                    xaxis_title="Terminal spot",
                    yaxis_title="P&L per share",
                )
                st.plotly_chart(apply_chart_layout(fig_payoff, 420), width="stretch")
            legs = pd.DataFrame(strategy.get("legs") or [])
            if not legs.empty:
                st.dataframe(
                    legs,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "contract": st.column_config.TextColumn("Contract"),
                        "type": st.column_config.TextColumn("Type"),
                        "expiration": st.column_config.TextColumn("Expiration"),
                        "dte": st.column_config.NumberColumn("DTE", format="%.0f"),
                        "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                        "quantity": st.column_config.NumberColumn("Qty", format="%.0f"),
                        "surface_iv": st.column_config.NumberColumn("Surface IV", format="%.2%"),
                        "pricing_iv": st.column_config.NumberColumn("Pricing IV", format="%.2%"),
                        "model_price": st.column_config.NumberColumn("Model Price", format="$%.2f"),
                        "market_price": st.column_config.NumberColumn("Market Price", format="$%.2f"),
                        "delta": st.column_config.NumberColumn("Delta", format="%.4f"),
                        "gamma": st.column_config.NumberColumn("Gamma", format="%.4f"),
                        "theta": st.column_config.NumberColumn("Theta/day", format="$%.4f"),
                        "vega": st.column_config.NumberColumn("Vega/1%", format="$%.4f"),
                    },
                )
            st.caption(
                f"Strategy source: {strategy.get('source', 'option chain plus fitted surface')}; "
                f"surface-priced legs {fmt_int(strategy.get('surface_priced_legs'))}/"
                f"{fmt_int(strategy.get('leg_count'))}; "
                f"breakevens {', '.join(fmt_money(value) for value in strategy.get('breakevens', [])) or 'n/a'}; "
                f"grid max profit {fmt_money(strategy.get('max_profit_100x'))}; "
                f"grid max loss {fmt_money(strategy.get('max_loss_100x'))}."
            )
            st.markdown('<div class="section-header">Strategy Scenario Engine</div>', unsafe_allow_html=True)
            scenario_cols = st.columns(4)
            with scenario_cols[0]:
                spot_span = st.slider("Spot shock range", 0.02, 0.25, 0.10, 0.01, format="%.2f")
            with scenario_cols[1]:
                max_days_passed = st.slider("Time decay days", 1, 60, 30, 1)
            with scenario_cols[2]:
                vol_span = st.slider("Vol shock range", 0.01, 0.20, 0.05, 0.01, format="%.2f")
            with scenario_cols[3]:
                skew_span = st.slider("Skew shock range", 0.00, 0.12, 0.03, 0.01, format="%.2f")
            spot_axis = tuple(round(x, 4) for x in np.linspace(-spot_span, spot_span, 5))
            time_axis = tuple(round(x, 2) for x in np.linspace(0, max_days_passed, 4))
            vol_axis = tuple(round(x, 4) for x in np.linspace(-vol_span, vol_span, 5))
            skew_axis = tuple(round(x, 4) for x in (-skew_span, 0.0, skew_span))
            scenarios = load_with_status(
                st,
                LoadingState(
                    title=f"{surface_symbol} {strategy_type} scenarios",
                    detail="Repricing strategy legs across spot, time, parallel-vol, and skew shocks.",
                    stage="strategy scenarios",
                    rows=5,
                ),
                lambda: get_strategy_scenarios_cached(
                    surface_symbol,
                    strategy_type,
                    spot_axis,
                    time_axis,
                    vol_axis,
                    skew_axis,
                    data_key,
                ),
            )
            if scenarios.get("available"):
                heatmap_points = pd.DataFrame(scenarios.get("spot_vol_heatmap") or [])
                if not heatmap_points.empty:
                    heatmap = heatmap_points.pivot_table(
                        index="vol_shift",
                        columns="spot_shift",
                        values="pnl_100x",
                        aggfunc="mean",
                    ).sort_index()
                    fig_strategy_scenario = go.Figure(
                        data=go.Heatmap(
                            z=heatmap.values,
                            x=[f"{value:.0%}" for value in heatmap.columns],
                            y=[f"{value:.0%}" for value in heatmap.index],
                            colorscale="RdYlGn",
                            zmid=0,
                            colorbar=dict(title="P&L"),
                        )
                    )
                    fig_strategy_scenario.update_layout(
                        title="Strategy P&L: Spot vs Vol",
                        xaxis_title="Spot shock",
                        yaxis_title="Vol shock",
                    )
                    st.plotly_chart(apply_chart_layout(fig_strategy_scenario, 420), width="stretch")
                scenario_points = pd.DataFrame(scenarios.get("points") or [])
                if not scenario_points.empty:
                    st.dataframe(
                        scenario_points.head(25),
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "spot_shift": st.column_config.NumberColumn("Spot Shift", format="%.2%"),
                            "time_pass_days": st.column_config.NumberColumn("Days Passed", format="%.0f"),
                            "vol_shift": st.column_config.NumberColumn("Vol Shift", format="%.2%"),
                            "skew_shift": st.column_config.NumberColumn("Skew Shift", format="%.2%"),
                            "shocked_spot": st.column_config.NumberColumn("Shocked Spot", format="$%.2f"),
                            "pnl_100x": st.column_config.NumberColumn("P&L", format="$%.2f"),
                        },
                    )
            else:
                st.markdown(
                    render_empty_state(
                        "Strategy scenarios unavailable",
                        scenarios.get("reason") or "Scenario repricing needs a priced strategy.",
                        "Refresh data or choose another strategy template.",
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                render_empty_state(
                    "Strategy unavailable",
                    strategy.get("reason") or "Template legs could not be selected from the current chain.",
                    "Refresh data, relax filters, or choose another template.",
                ),
                unsafe_allow_html=True,
            )

    with risk_tab:
        st.markdown('<div class="section-header">Portfolio And Cross-Asset Risk</div>', unsafe_allow_html=True)
        portfolio_upload = st.file_uploader(
            "Portfolio CSV",
            type=["csv"],
            help=CONTROL_HELP["portfolio_csv"],
        )
        portfolio_bytes = portfolio_upload.getvalue() if portfolio_upload is not None else None
        portfolio = load_with_status(
            st,
            LoadingState(
                title="Portfolio risk",
                detail="Matching uploaded CSV positions to option-chain contracts and aggregating Greeks.",
                stage="portfolio",
                rows=4,
            ),
            lambda: get_portfolio_metrics_cached(portfolio_bytes, data_key),
        )
        if not portfolio.get("configured"):
            st.markdown(
                render_empty_state(
                    "Portfolio book unavailable",
                    "No configured positions. Portfolio P&L, VaR, Sharpe, and drawdown remain disabled.",
                    "Upload a CSV with symbol, expiry, strike, type, quantity, and cost columns.",
                ),
                unsafe_allow_html=True,
            )
        elif portfolio.get("available"):
            totals = portfolio.get("totals") or {}
            portfolio_cols = st.columns(5)
            portfolio_metrics = [
                ("Market Value", fmt_money(totals.get("market_value_100x")), "100x"),
                ("Unrealized P&L", fmt_money(totals.get("unrealized_pnl_100x")), "100x"),
                ("Delta", _fmt_number(totals.get("delta_100x")), "100x"),
                ("Theta/day", fmt_money(totals.get("theta_100x")), "100x"),
                ("Vega/1%", fmt_money(totals.get("vega_100x")), "100x"),
            ]
            for col, (label, value, delta) in zip(portfolio_cols, portfolio_metrics):
                with col:
                    st.metric(label, value, delta=delta)
            positions_frame = pd.DataFrame(portfolio.get("positions") or [])
            if not positions_frame.empty:
                st.dataframe(
                    positions_frame,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "symbol": st.column_config.TextColumn("Symbol"),
                        "contract": st.column_config.TextColumn("Contract"),
                        "type": st.column_config.TextColumn("Type"),
                        "expiration": st.column_config.TextColumn("Expiration"),
                        "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                        "quantity": st.column_config.NumberColumn("Qty", format="%.0f"),
                        "cost": st.column_config.NumberColumn("Cost", format="$%.2f"),
                        "model_price": st.column_config.NumberColumn("Model Price", format="$%.2f"),
                        "market_value_100x": st.column_config.NumberColumn("Market Value", format="$%.2f"),
                        "unrealized_pnl_100x": st.column_config.NumberColumn("P&L", format="$%.2f"),
                        "delta": st.column_config.NumberColumn("Delta", format="%.4f"),
                        "theta": st.column_config.NumberColumn("Theta/day", format="$%.4f"),
                        "vega": st.column_config.NumberColumn("Vega/1%", format="$%.4f"),
                    },
                )
            scenario_frame = pd.DataFrame(portfolio.get("scenario_pnl") or [])
            if not scenario_frame.empty:
                scenario_grid = scenario_frame.pivot_table(
                    index="vol_shift",
                    columns="spot_shift",
                    values="pnl_100x",
                    aggfunc="mean",
                ).sort_index()
                fig_portfolio_scenario = go.Figure(
                    data=go.Heatmap(
                        z=scenario_grid.values,
                        x=[f"{value:.0%}" for value in scenario_grid.columns],
                        y=[f"{value:.0%}" for value in scenario_grid.index],
                        colorscale="RdYlGn",
                        zmid=0,
                        colorbar=dict(title="P&L"),
                    )
                )
                fig_portfolio_scenario.update_layout(
                    title="Portfolio P&L: Spot vs Vol",
                    xaxis_title="Spot shock",
                    yaxis_title="Vol shock",
                )
                st.plotly_chart(apply_chart_layout(fig_portfolio_scenario, 420), width="stretch")
            opt_cols = st.columns([2, 1])
            with opt_cols[0]:
                hedge_objective = st.selectbox(
                    "Hedge objective",
                    ["delta-neutral", "vega-neutral", "theta target", "max loss constraint"],
                    index=0,
                    help=CONTROL_HELP["hedge_objective"],
                )
            with opt_cols[1]:
                theta_target = st.number_input("Theta target", value=0.0, step=10.0)
            optimization = get_portfolio_optimization_cached(
                portfolio_bytes,
                hedge_objective,
                float(theta_target),
                data_key,
            )
            suggestions = pd.DataFrame(optimization.get("suggestions") or [])
            if not suggestions.empty:
                st.dataframe(
                    suggestions,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "contract": st.column_config.TextColumn("Contract"),
                        "symbol": st.column_config.TextColumn("Symbol"),
                        "size": st.column_config.NumberColumn("Size", format="%.0f"),
                        "estimated_cost": st.column_config.NumberColumn("Estimated Cost", format="$%.2f"),
                        "post_trade_exposure": st.column_config.NumberColumn("Post Trade Exposure", format="%.2f"),
                        "residual_exposure": st.column_config.NumberColumn("Residual", format="%.2f"),
                        "trade_offs": st.column_config.TextColumn("Trade-offs"),
                    },
                )
            unmatched = pd.DataFrame(portfolio.get("unmatched") or [])
            if not unmatched.empty:
                st.warning(f"{len(unmatched)} uploaded position(s) could not be matched.")
        else:
            st.markdown(
                render_empty_state(
                    "Portfolio import unavailable",
                    portfolio.get("reason") or "Uploaded CSV could not be parsed or matched.",
                    "Check required columns and option identifiers.",
                ),
                unsafe_allow_html=True,
            )

        surface_shocks = surface_meta.get("surface_shocks") or {}
        if surface_shocks.get("available"):
            st.markdown('<div class="section-header">Surface Shock Analysis</div>', unsafe_allow_html=True)
            st.caption(
                f"Scenario source: {surface_shocks.get('source', 'current option chain')}; "
                f"assumption: {surface_shocks.get('position_assumption', 'one long contract per option row')}; "
                f"contracts: {fmt_int(surface_shocks.get('base_contracts'))}; "
                f"base market value: {fmt_money(surface_shocks.get('base_market_value'))}; "
                f"base delta: {_fmt_number(surface_shocks.get('base_delta'))}; "
                f"base vega/1%: {_fmt_number(surface_shocks.get('base_vega'))}."
            )
            shock_display = pd.DataFrame(surface_shocks.get("scenarios") or [])
            if not shock_display.empty:
                st.dataframe(
                    shock_display,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "scenario": st.column_config.TextColumn("Scenario"),
                        "spot_shift": st.column_config.NumberColumn("Spot Shift", format="%.2%"),
                        "vol_shift": st.column_config.NumberColumn("Vol Shift", format="%.2%"),
                        "contracts": st.column_config.NumberColumn("Contracts", format="%d"),
                        "unit_contract_pnl": st.column_config.NumberColumn("Unit Basket P&L", format="$%.2f"),
                        "mean_contract_pnl": st.column_config.NumberColumn("Mean P&L", format="$%.2f"),
                        "max_contract_loss": st.column_config.NumberColumn("Worst Contract", format="$%.2f"),
                        "max_contract_gain": st.column_config.NumberColumn("Best Contract", format="$%.2f"),
                        "delta_before": st.column_config.NumberColumn("Delta Before", format="%.3f"),
                        "delta_after": st.column_config.NumberColumn("Delta After", format="%.3f"),
                        "delta_change": st.column_config.NumberColumn("Delta Change", format="%.3f"),
                        "vega_before": st.column_config.NumberColumn("Vega Before", format="%.3f"),
                        "vega_after": st.column_config.NumberColumn("Vega After", format="%.3f"),
                        "vega_change": st.column_config.NumberColumn("Vega Change", format="%.3f"),
                        "mean_shocked_iv": st.column_config.NumberColumn("Mean Shocked IV", format="%.2%"),
                    },
                )
        else:
            st.markdown(
                render_empty_state(
                    "Surface shocks unavailable",
                    surface_shocks.get("reason")
                    or "No option rows have usable price, IV, strike, and expiry inputs for scenario repricing.",
                    "Refresh data, relax filters, or select a symbol with a deeper options chain.",
                ),
                unsafe_allow_html=True,
            )

        if show_correlations and len(selected_symbols) > 1:
            corr = load_with_status(
                st,
                LoadingState(
                    title="Realized correlation matrix",
                    detail="Fetching yfinance historical closes for the selected universe and calculating aligned returns.",
                    stage="correlation",
                    rows=max(3, min(len(selected_symbols), 8)),
                ),
                lambda: get_correlation_matrix_cached(tuple(sorted(selected_symbols))),
            )
            if not corr.empty:
                fig_corr = go.Figure(
                    data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorscale="RdBu",
                        zmid=0,
                        zmin=-1,
                        zmax=1,
                        text=corr.round(2).values,
                        texttemplate="%{text}",
                        colorbar=dict(title="Return corr"),
                    )
                )
                fig_corr.update_layout(title="6M Realized Return Correlation")
                st.plotly_chart(apply_chart_layout(fig_corr, 460), width="stretch")
            else:
                st.markdown(
                    render_empty_state(
                        "Correlation matrix unavailable",
                        "Fewer than two selected symbols returned at least 20 aligned daily returns.",
                        "Refresh data, reduce the universe, or choose symbols with longer yfinance history.",
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                render_empty_state(
                    "Correlation panel idle",
                    "Select at least two symbols and enable realized correlation to compute this panel.",
                ),
                unsafe_allow_html=True,
            )

    with diagnostics_tab:
        st.markdown('<div class="section-header">Diagnostics And Data Provenance</div>', unsafe_allow_html=True)
        health = connector.get_system_health()
        with card(st, title="System Health", kicker="Diagnostics", actions=["R", "i"]):
            col1, col2 = st.columns(2)
            with col1:
                st.json(health.get("overall", {}))
            with col2:
                st.json(health.get("data_contract", {}))

        report_cols = st.columns([1, 1, 1])
        with report_cols[0]:
            if st.button("Generate research report", width="stretch"):
                report = connector.generate_research_report(surface_symbol)
                if report.get("available"):
                    st.success(f"Report written to {report.get('path')}")
                else:
                    st.error(report.get("reason", "Report generation failed"))
        with report_cols[1]:
            if st.button("Async refresh", width="stretch"):
                st.json(connector.request_async_refresh(surface_symbol))
        with report_cols[2]:
            st.json(connector.get_async_refresh_status(surface_symbol))

        st.markdown("#### Advanced Research Modules")
        research_cols = st.columns(4)
        with research_cols[0]:
            st.json(connector.get_ml_anomaly_detector(surface_symbol))
        with research_cols[1]:
            st.json(connector.get_vol_regime_classifier(surface_symbol))
        with research_cols[2]:
            st.json(connector.get_forecasting_module(surface_symbol))
        with research_cols[3]:
            st.json(connector.get_news_event_overlay(surface_symbol))

        st.markdown("#### Page State")
        st.json(
            {
                "state": dashboard_state.snapshot(),
                "pages": [{"key": page.key, "title": page.title, "workflow": page.workflow} for page in page_registry],
            }
        )

        st.markdown("#### Market Calendar")
        st.json({k: str(v) if isinstance(v, datetime) else v for k, v in market_status.items()})

        st.markdown("#### Surface Alerts")
        alert_cols = st.columns(5)
        with alert_cols[0]:
            iv_rank_threshold = st.slider("IV rank", 0.0, 1.0, 0.80, 0.05)
        with alert_cols[1]:
            skew_threshold = st.slider("Skew", 0.00, 0.20, 0.05, 0.01)
        with alert_cols[2]:
            fit_threshold = st.slider("Fit error", 0.00, 0.20, 0.03, 0.01)
        with alert_cols[3]:
            stale_threshold = st.slider("Stale min", 1, 240, 30, 1)
        with alert_cols[4]:
            residual_threshold = st.slider("Residual", 0.00, 0.50, 0.10, 0.01)
        alerts_payload = get_surface_alerts_cached(
            surface_symbol,
            (
                float(iv_rank_threshold),
                float(skew_threshold),
                float(fit_threshold),
                float(stale_threshold),
                float(residual_threshold),
            ),
            data_key,
        )
        alerts_frame = pd.DataFrame(alerts_payload.get("alerts") or [])
        st.caption(
            f"Alert source: {alerts_payload.get('source', 'local rules')}; "
            f"logged to {alerts_payload.get('log_path') or 'disabled'}; "
            f"active alerts {fmt_int(alerts_payload.get('alert_count'))}."
        )
        if alerts_frame.empty:
            st.success("No configured alerts are active.")
        else:
            st.dataframe(alerts_frame, width="stretch", hide_index=True)

        st.markdown("#### Latest Surface Metadata")
        st.json({k: str(v) if isinstance(v, datetime) else v for k, v in surface_meta.items()})
        expiry_quality = surface_meta.get("expiry_quality") or {}
        if expiry_quality:
            quality_rows = []
            for expiry, payload in sorted(expiry_quality.items()):
                buckets = payload.get("reason_buckets") or {}
                quality_rows.append(
                    {
                        "Expiry": expiry,
                        "Score": payload.get("score"),
                        "Raw Quotes": payload.get("raw_quotes"),
                        "Valid Quotes": payload.get("valid_quotes"),
                        "Rejected Quotes": payload.get("rejected_quotes"),
                        "Surface Quotes": payload.get("surface_quotes"),
                        "Reason Buckets": ", ".join(
                            f"{reason}: {count}" for reason, count in sorted(buckets.items()) if count
                        )
                        or "none",
                    }
                )
            st.markdown("#### Expiry Data Quality")
            st.dataframe(
                pd.DataFrame(quality_rows),
                width="stretch",
                hide_index=True,
                column_config={
                    "Expiry": st.column_config.TextColumn("Expiry"),
                    "Score": st.column_config.NumberColumn("Score", format="%.1f"),
                    "Raw Quotes": st.column_config.NumberColumn("Raw Quotes", format="%d"),
                    "Valid Quotes": st.column_config.NumberColumn("Valid Quotes", format="%d"),
                    "Rejected Quotes": st.column_config.NumberColumn("Rejected Quotes", format="%d"),
                    "Surface Quotes": st.column_config.NumberColumn("Surface Quotes", format="%d"),
                    "Reason Buckets": st.column_config.TextColumn("Reason Buckets"),
                },
            )

        surface_quality = surface_meta.get("surface_quality") or {}
        if surface_quality:
            st.markdown("#### Surface Quality")
            st.json(surface_quality)
        warnings_list = surface_meta.get("warnings") or []
        if warnings_list:
            st.warning(" | ".join(str(item) for item in warnings_list[:4]))

    with report_tab:
        st.markdown('<div class="section-header">Report Export Panel</div>', unsafe_allow_html=True)
        export_payload = _analysis_export_payload()
        report_cols = st.columns(3)
        with report_cols[0]:
            if st.button("Export HTML report", width="stretch", key="report_panel_html"):
                report = connector.generate_research_report(surface_symbol)
                if report.get("available"):
                    st.success(f"HTML report written to {report.get('path')}")
                else:
                    st.error(report.get("reason", "HTML report generation failed"))
        with report_cols[1]:
            notebook_path = f"reports/{surface_symbol}_surface_analysis.ipynb"
            export_notebook = getattr(connector, "export_analysis_notebook", None)
            if st.button("Export notebook", width="stretch", key="report_panel_notebook"):
                if export_notebook is None:
                    st.error("Notebook export is unavailable for this connector.")
                else:
                    notebook = export_notebook(export_payload, notebook_path)
                    if notebook.get("available"):
                        st.success(f"Notebook written to {notebook.get('path')}")
                    else:
                        st.error(notebook.get("reason", "Notebook export failed"))
        with report_cols[2]:
            save_workspace = getattr(connector, "save_workspace", None)
            if st.button("Save workspace", width="stretch", key="report_panel_workspace"):
                if save_workspace is None:
                    st.error("Workspace export is unavailable for this connector.")
                else:
                    workspace = {
                        "name": f"{surface_symbol} Phase 6 workspace",
                        "selected_symbols": selected_symbols,
                        "filters": {
                            "min_open_interest": min_open_interest,
                            "min_volume": min_volume,
                            "max_bid_ask_spread_pct": max_spread_pct,
                            "max_quote_age_days": max_quote_age_days,
                            "surface_x_axis": surface_x_axis,
                            "selected_fit_mode": fit_mode_view["selected_mode"],
                        },
                        "model_settings": {
                            "option_price_source": option_price_source,
                            "pricing_model": pricing_model,
                            "fit_preset": fit_preset,
                            "fit_max_bid_ask_spread_pct": fit_max_spread_pct,
                            "fit_max_quote_age_days": fit_max_quote_age_days,
                            "fit_min_volume": fit_min_volume,
                            "fit_min_open_interest": fit_min_open_interest,
                            "fit_moneyness": list(fit_moneyness_band),
                            "fit_max_raw_iv": fit_max_raw_iv,
                            "fit_no_arbitrage_policy": fit_no_arbitrage_policy,
                            "fit_last_only_policy": fit_last_only_policy,
                        },
                        "chart_layout": {
                            "show_3d_surface": show_3d_surface,
                            "show_chain": show_chain,
                            "show_reliability_overlay": show_reliability_overlay,
                        },
                        "provenance": export_payload["provenance"],
                    }
                    saved = save_workspace(workspace, name=f"{surface_symbol}_phase6_workspace")
                    if saved.get("available"):
                        st.success(f"Workspace written to {saved.get('path')}")
                    else:
                        st.error(saved.get("reason", "Workspace export failed"))
        diagnostics_payload = fit_diagnostics_export_payload(surface_symbol, surface_meta)
        diagnostics_frame = pd.DataFrame(diagnostics_payload.get("row_weights") or [])
        export_cols = st.columns(2)
        with export_cols[0]:
            st.download_button(
                "Export fit diagnostics JSON",
                json.dumps(diagnostics_payload, default=str, indent=2).encode("utf-8"),
                file_name=f"{surface_symbol}_fit_diagnostics.json",
                mime="application/json",
                key="fit_diagnostics_export_json",
                width="stretch",
            )
        with export_cols[1]:
            if diagnostics_frame.empty:
                st.download_button(
                    "Export row diagnostics CSV",
                    b"",
                    file_name=f"{surface_symbol}_row_diagnostics.csv",
                    mime="text/csv",
                    key="fit_diagnostics_export_csv_empty",
                    width="stretch",
                    disabled=True,
                )
            else:
                st.download_button(
                    "Export row diagnostics CSV",
                    dataframe_to_csv_bytes(diagnostics_frame),
                    file_name=f"{surface_symbol}_row_diagnostics.csv",
                    mime="text/csv",
                    key="fit_diagnostics_export_csv",
                    width="stretch",
                )
        st.caption(
            f"Export payload source {export_payload['provenance'].get('surface_source') or 'unknown'}; "
            f"mode {export_payload['provenance'].get('surface_mode') or 'unknown'}; "
            f"data timestamp {export_payload.get('data_timestamp')}; "
            f"model assumptions {export_payload.get('model_assumptions') or 'n/a'}."
        )
        st.json(export_payload)
        list_workspaces = getattr(connector, "list_workspaces", None)
        if list_workspaces is not None:
            saved_workspaces = list_workspaces()
            if saved_workspaces:
                st.markdown("#### Saved Workspaces")
                st.dataframe(pd.DataFrame(saved_workspaces[:5]), width="stretch", hide_index=True)

    st.markdown(
        f"""
    <div class="small-note">
        Status row generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
        This app is for research and education, not investment advice.
    </div>
    """,
        unsafe_allow_html=True,
    )

    if auto_refresh and system_ready:
        time.sleep(refresh_interval)
        st.rerun()
