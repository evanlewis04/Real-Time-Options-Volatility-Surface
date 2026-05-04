from __future__ import annotations


def run_dashboard() -> None:

    import os
    import sys
    import time
    import warnings
    from datetime import datetime
    from typing import Any, Dict, Iterable, Tuple

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    from src.dashboard.formatting import fmt_int, fmt_money, fmt_pct
    from src.dashboard.loading import LoadingState, load_with_status, render_empty_state
    from src.dashboard.surface_view import extract_smile, surface_mesh, surface_stats
    from src.dashboard.tables import dataframe_to_csv_bytes, filter_market_snapshot, filter_option_chain
    from src.dashboard.theme import apply_chart_layout, inject_theme, status_pill
    from src.dashboard.tooltips import COLUMN_HELP, CONTROL_HELP, KPI_HELP

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

        def get_current_data(self, symbol: str) -> Dict[str, Any]:
            prices = {"AAPL": 196.50, "MSFT": 416.50, "NVDA": 138.50, "TSLA": 325.00, "SPY": 578.00}
            vols = {"AAPL": 0.25, "MSFT": 0.22, "NVDA": 0.40, "TSLA": 0.50, "SPY": 0.15}
            spot = prices.get(symbol.upper(), 100.0)
            iv = vols.get(symbol.upper(), 0.30)
            return {
                "symbol": symbol.upper(),
                "price": spot,
                "price_source": "static fallback",
                "data_mode": "Fallback",
                "iv_source": "static fallback",
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
                "mode": "Fallback",
                "surface_mode": "Fallback",
                "surface_source": "static local fallback",
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
                    "reason_buckets": {},
                    "expiries": {},
                },
                "fallback_reason": "DashboardConnector could not be imported",
                "surface_points": 96,
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
                source="static local fallback",
                fallback_reason="DashboardConnector could not be imported",
                mode="Fallback",
            )

        def get_portfolio_metrics(self):
            return {"configured": False, "message": "No position book configured"}

        def get_correlation_matrix(self, symbols: Iterable[str], period: str = "6mo"):
            return pd.DataFrame()

        def get_historical_metrics(self, symbol: str, period: str = "1y"):
            return {"available": False, "reason": "No historical provider in fallback mode"}

        def get_market_status(self):
            from src.data.market_calendar import MarketCalendar

            return MarketCalendar().status(self.timestamp).as_dict()

        def configure_liquidity_filters(self, **kwargs):
            return {}

        def configure_option_price_source(self, price_source: str):
            return price_source

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
            }

        def trigger_data_refresh(self):
            self.timestamp = datetime.now()
            return {"status": "success", "message": "Fallback timestamp refreshed"}

        def start_real_time_updates(self):
            return False


    @st.cache_resource
    def init_dashboard_system():
        if CONNECTOR_AVAILABLE:
            connector = DashboardConnector()
            if REAL_SYSTEM_AVAILABLE:
                connector.start_real_time_updates()
            return connector, True
        return MinimalFallbackConnector(), False


    connector, system_ready = init_dashboard_system()


    @st.cache_data(ttl=120, show_spinner=False)
    def get_current_data_cached(symbol: str, data_key: Tuple[int, int, float, int, str]):
        return connector.get_current_data(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_vol_surface_data_cached(symbol: str, data_key: Tuple[int, int, float, int, str]):
        return connector.get_vol_surface_data(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_surface_metadata_cached(symbol: str, data_key: Tuple[int, int, float, int, str]):
        return connector.get_surface_metadata(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_options_chain_cached(symbol: str, data_key: Tuple[int, int, float, int, str]):
        return connector.get_options_chain_snapshot(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_market_snapshot_cached(symbol: str, data_key: Tuple[int, int, float, int, str]):
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

    with st.sidebar:
        st.markdown("### Workspace")
        selected_symbols = st.multiselect(
            "Universe",
            options=available_symbols,
            default=["AAPL", "MSFT", "TSLA", "NVDA", "SPY"],
            help=CONTROL_HELP["universe"],
        )
        show_3d_surface = st.checkbox("3D surface", value=True, help=CONTROL_HELP["show_3d_surface"])
        show_correlations = st.checkbox("Realized correlation", value=True, help=CONTROL_HELP["show_correlations"])
        show_chain = st.checkbox("Option chain", value=True, help=CONTROL_HELP["show_chain"])
        auto_refresh = st.checkbox("Auto refresh", value=False, help=CONTROL_HELP["auto_refresh"])
        refresh_interval = st.slider(
            "Refresh interval seconds",
            15,
            180,
            60,
            15,
            help=CONTROL_HELP["refresh_interval"],
        )
        st.markdown("### Data Filters")
        max_spread_pct = st.slider(
            "Max spread percent",
            0.05,
            1.50,
            0.75,
            0.05,
            help=CONTROL_HELP["max_spread_pct"],
        )
        min_open_interest = st.number_input(
            "Min open interest",
            min_value=0,
            value=0,
            step=10,
            help=CONTROL_HELP["min_open_interest"],
        )
        min_volume = st.number_input(
            "Min volume",
            min_value=0,
            value=0,
            step=5,
            help=CONTROL_HELP["min_volume"],
        )
        max_quote_age_days = st.number_input(
            "Max quote age days",
            min_value=0,
            value=5,
            step=1,
            help=CONTROL_HELP["max_quote_age_days"],
        )
        option_price_source = st.selectbox(
            "IV price source",
            options=["mark", "midpoint", "last"],
            index=0,
            help=CONTROL_HELP["option_price_source"],
        )
        if st.button("Refresh data", width="stretch"):
            result = connector.trigger_data_refresh()
            st.cache_data.clear()
            if result.get("status") == "success":
                st.success(result.get("message", "Data refreshed"))
            else:
                st.error(result.get("message", "Refresh failed"))

    if not selected_symbols:
        st.warning("Select at least one symbol from the sidebar to begin analysis.")
        st.stop()

    surface_symbol = st.selectbox(
        "Primary underlying",
        selected_symbols,
        index=0,
        help=CONTROL_HELP["primary_underlying"],
    )
    data_key = (
        int(min_open_interest),
        int(min_volume),
        float(max_spread_pct),
        int(max_quote_age_days),
        str(option_price_source),
    )
    connector.configure_liquidity_filters(
        min_open_interest=data_key[0],
        min_volume=data_key[1],
        max_bid_ask_spread_pct=data_key[2],
        max_quote_age_days=data_key[3],
    )
    connector.configure_option_price_source(data_key[4])

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
    stats = surface_stats(strikes, expiries, vol_surface, current_data["price"])
    market_status = get_market_status_cached()

    surface_mode = surface_meta.get("surface_mode") or current_data.get("data_mode", "Unknown")
    source_label = surface_meta.get("surface_source") or current_data.get("price_source", "Unknown")
    quality_score = surface_meta.get("surface_quality_score") or surface_meta.get("data_quality_score")
    reason_buckets = surface_meta.get("quality_reason_buckets") or surface_meta.get("rejection_reasons") or {}
    reason_bucket_text = ", ".join(
        f"{reason}: {fmt_int(count)}" for reason, count in sorted(reason_buckets.items()) if count
    ) or "none"

    st.markdown(
        f"""
    <div class="workstation-header">
        <div class="workstation-title">Options Volatility Surface Workstation</div>
        <div class="workstation-subtitle">
            {surface_symbol} | Spot {fmt_money(current_data.get("price"))} |
            Source {source_label} | Market {market_status.get("session_state", "Unknown")} |
            Updated {datetime.now().strftime("%H:%M:%S")}
        </div>
        <div style="margin-top:0.55rem;">
            {status_pill("Surface", surface_mode)}
            {status_pill("Price", current_data.get("data_mode", "Unknown"))}
            {status_pill("IV", current_data.get("iv_source", "Unknown"))}
            {status_pill("Market", market_status.get("session_state", "Unknown"))}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(6)
    kpis = [
        ("Spot", fmt_money(current_data.get("price")), current_data.get("price_source", "")),
        ("ATM IV", fmt_pct(stats.get("atm_iv")), "nearest strike"),
        ("Term Spread", fmt_pct(stats.get("term_spread")), "back minus front"),
        ("Surface Points", fmt_int(stats.get("points")), surface_mode),
        ("Contracts", fmt_int(surface_meta.get("valid_rows") or current_data.get("contracts")), "valid rows"),
        ("Quality Score", f"{quality_score:.1f}/100" if quality_score is not None else "n/a", "surface"),
    ]
    for col, (label, value, delta) in zip(kpi_cols, kpis):
        with col:
            st.metric(label, value, delta=delta if delta else None, help=KPI_HELP.get(label))

    st.markdown(
        f"""
    <div class="quality-row">
        <strong>Data quality:</strong>
        score {f"{quality_score:.1f}/100" if quality_score is not None else "n/a"};
        raw rows {fmt_int(surface_meta.get("raw_rows"))};
        valid rows {fmt_int(surface_meta.get("valid_rows"))};
        rejected rows {fmt_int(surface_meta.get("rejected_rows"))};
        reason buckets {reason_bucket_text};
        liquidity rejects {fmt_int(surface_meta.get("liquidity_filtered_count"))};
        cache age {fmt_int(surface_meta.get("cache_age_seconds"))}s;
        market {market_status.get("reason", "unknown")};
        delay {fmt_int(market_status.get("data_delay_minutes"))} min;
        fallback reason {surface_meta.get("fallback_reason") or "none"}.
        risk-free curve {surface_meta.get("risk_free_rate_source") or "unknown"};
        30D rate {fmt_pct(surface_meta.get("risk_free_rate_30d"))};
        dividend source {surface_meta.get("dividend_source") or "unknown"};
        30D dividend yield {fmt_pct(surface_meta.get("effective_dividend_yield_30d"))};
        corporate actions {fmt_int(surface_meta.get("corporate_action_warning_count"))} warning(s).
        stale quotes {fmt_int(surface_meta.get("stale_quote_count"))};
        last-only quotes {fmt_int(surface_meta.get("last_only_quote_count"))};
        crossed/locked markets {fmt_int(surface_meta.get("crossed_locked_rejected_count"))};
        parity violations {fmt_int(surface_meta.get("parity_violation_count"))};
        filters OI >= {fmt_int(surface_meta.get("min_open_interest"))},
        volume >= {fmt_int(surface_meta.get("min_volume"))},
        spread <= {fmt_pct(surface_meta.get("max_bid_ask_spread_pct"), 0)},
        quote age <= {fmt_int(surface_meta.get("max_quote_age_days"))}d;
        IV price source {surface_meta.get("option_price_source") or "mark"};
        computed IV {fmt_int(surface_meta.get("computed_iv_count"))}.
    </div>
    """,
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

    surface_tab, chain_tab, skew_tab, risk_tab, diagnostics_tab = st.tabs(
        ["Surface", "Chain", "Skew & Term", "Risk", "Diagnostics"]
    )

    with surface_tab:
        st.markdown('<div class="section-header">Volatility Surface</div>', unsafe_allow_html=True)
        strike_mesh, expiry_mesh, vols = surface_mesh(strikes, expiries, vol_surface)

        if show_3d_surface:
            fig_3d = go.Figure(
                data=[
                    go.Surface(
                        z=vols,
                        x=strike_mesh,
                        y=expiry_mesh,
                        colorscale="Cividis",
                        colorbar=dict(title="IV"),
                        hovertemplate="Strike: $%{x:.2f}<br>DTE: %{y:.0f}<br>IV: %{z:.2%}<extra></extra>",
                    )
                ]
            )
            fig_3d.update_layout(
                title=f"{surface_symbol} Implied Volatility Surface",
                scene=dict(
                    xaxis_title="Strike",
                    yaxis_title="Days to expiry",
                    zaxis_title="Annualized IV",
                    camera=dict(eye=dict(x=1.45, y=1.35, z=1.0)),
                ),
                margin=dict(l=0, r=0, t=45, b=0),
                height=620,
            )
            st.plotly_chart(fig_3d, width="stretch")

        fig_heatmap = go.Figure(
            data=[
                go.Heatmap(
                    z=vols,
                    x=strike_mesh[0, :] if strike_mesh.ndim == 2 else strike_mesh,
                    y=expiry_mesh[:, 0] if expiry_mesh.ndim == 2 else expiry_mesh,
                    colorscale="RdBu_r",
                    colorbar=dict(title="IV"),
                    hovertemplate="Strike: $%{x:.2f}<br>DTE: %{y:.0f}<br>IV: %{z:.2%}<extra></extra>",
                )
            ]
        )
        fig_heatmap.update_layout(
            title=f"{surface_symbol} Surface Heatmap",
            xaxis_title="Strike",
            yaxis_title="Days to expiry",
        )
        st.plotly_chart(apply_chart_layout(fig_heatmap, 430), width="stretch")

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
                "parityViolation",
                "parityError",
                "riskFreeRate",
                "effectiveDividendYield",
                "discreteDividendAmount",
                "quoteQuality",
                "isCrossedMarket",
                "isLockedMarket",
                "quoteAgeSeconds",
                "bidAskSpreadPct",
            ]
            display_cols = [col for col in display_cols if col in filtered.columns]
            st.download_button(
                "Export chain CSV",
                dataframe_to_csv_bytes(filtered[display_cols]),
                file_name=f"{surface_symbol}_option_chain.csv",
                mime="text/csv",
            )
            st.dataframe(
                filtered[display_cols],
                width="stretch",
                hide_index=True,
                column_config={
                    "type": st.column_config.TextColumn("Type", help=COLUMN_HELP["type"]),
                    "expiration": st.column_config.DateColumn("Expiration", help=COLUMN_HELP["expiration"]),
                    "daysToExpiration": st.column_config.NumberColumn(
                        "DTE",
                        format="%d",
                        help=COLUMN_HELP["daysToExpiration"],
                    ),
                    "strike": st.column_config.NumberColumn(format="$%.2f", help=COLUMN_HELP["strike"]),
                    "moneyness": st.column_config.NumberColumn(format="%.3f", help=COLUMN_HELP["moneyness"]),
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
                    "parityViolation": st.column_config.CheckboxColumn(
                        "Parity Flag",
                        help=COLUMN_HELP["parityViolation"],
                    ),
                    "parityError": st.column_config.NumberColumn(
                        "Parity Error",
                        format="$%.2f",
                        help=COLUMN_HELP["parityError"],
                    ),
                    "riskFreeRate": st.column_config.NumberColumn(
                        "Rate",
                        format="%.2%",
                        help=COLUMN_HELP["riskFreeRate"],
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
                },
            )
            st.caption(
                f"Showing {len(filtered):,} of {len(chain_df):,} valid contracts. "
                f"Source: {chain_meta.get('source', 'unknown')}; mode: {chain_meta.get('mode', 'unknown')}; "
                f"liquidity rejects: {chain_meta.get('liquidity_filtered_count', 0):,}."
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
        smile_strikes, smile_vols, smile_days = extract_smile(strikes, expiries, vol_surface)
        col1, col2 = st.columns(2)
        with col1:
            fig_smile = go.Figure()
            fig_smile.add_trace(
                go.Scatter(
                    x=smile_strikes,
                    y=smile_vols,
                    mode="lines+markers",
                    name=f"{smile_days:.0f} DTE",
                    line=dict(color="#1f7a8c", width=3),
                    hovertemplate="Strike: $%{x:.2f}<br>IV: %{y:.2%}<extra></extra>",
                )
            )
            fig_smile.add_vline(x=current_data["price"], line_width=1, line_dash="dash", line_color="#667085")
            fig_smile.update_layout(title=f"{surface_symbol} Front Smile", xaxis_title="Strike", yaxis_title="Annualized IV")
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
            fig_term.update_layout(title=f"{surface_symbol} ATM Term Structure", xaxis_title="Days to expiry", yaxis_title="Annualized IV")
            st.plotly_chart(apply_chart_layout(fig_term, 430), width="stretch")

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
        else:
            st.markdown(
                render_empty_state(
                    "Historical volatility unavailable",
                    f"Realized-vol fetch did not return enough usable closes: {hist_metrics.get('reason', 'unknown reason')}.",
                    "Refresh data or switch to a symbol with liquid yfinance history.",
                ),
                unsafe_allow_html=True,
            )

    with risk_tab:
        st.markdown('<div class="section-header">Portfolio And Cross-Asset Risk</div>', unsafe_allow_html=True)
        portfolio = connector.get_portfolio_metrics()
        if not portfolio.get("configured"):
            st.markdown(
                render_empty_state(
                    "Portfolio book unavailable",
                    "No configured positions. Portfolio P&L, VaR, Sharpe, and drawdown remain disabled.",
                    "Use realized correlations here until position import is added.",
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
        col1, col2 = st.columns(2)
        with col1:
            st.json(health.get("overall", {}))
        with col2:
            st.json(health.get("data_contract", {}))

        st.markdown("#### Market Calendar")
        st.json({k: str(v) if isinstance(v, datetime) else v for k, v in market_status.items()})

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
            st.dataframe(pd.DataFrame(quality_rows), width="stretch", hide_index=True)

        surface_quality = surface_meta.get("surface_quality") or {}
        if surface_quality:
            st.markdown("#### Surface Quality")
            st.json(surface_quality)
        warnings_list = surface_meta.get("warnings") or []
        if warnings_list:
            st.warning(" | ".join(str(item) for item in warnings_list[:4]))

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
