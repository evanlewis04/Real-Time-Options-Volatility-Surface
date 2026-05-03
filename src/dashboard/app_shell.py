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
    from src.dashboard.surface_view import extract_smile, surface_mesh, surface_stats
    from src.dashboard.theme import apply_chart_layout, inject_theme, status_pill

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
                "bid_ask_spread": None,
                "contracts": None,
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
                "fallback_reason": "DashboardConnector could not be imported",
                "surface_points": 96,
            }

        def get_options_chain_snapshot(self, symbol: str):
            return pd.DataFrame(), self.get_surface_metadata(symbol)

        def get_portfolio_metrics(self):
            return {"configured": False, "message": "No position book configured"}

        def get_correlation_matrix(self, symbols: Iterable[str], period: str = "6mo"):
            return pd.DataFrame()

        def get_historical_metrics(self, symbol: str, period: str = "1y"):
            return {"available": False, "reason": "No historical provider in fallback mode"}

        def get_system_health(self):
            return {
                "overall": {"yfinance_available": False, "last_update": self.timestamp, "option_chain_cache_entries": 0},
                "performance": {"real_time_active": False, "update_interval": 30},
                "data_contract": {"price_provider": "static fallback"},
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
    def get_current_data_cached(symbol: str):
        return connector.get_current_data(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_vol_surface_data_cached(symbol: str):
        return connector.get_vol_surface_data(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_surface_metadata_cached(symbol: str):
        return connector.get_surface_metadata(symbol)


    @st.cache_data(ttl=300, show_spinner=False)
    def get_options_chain_cached(symbol: str):
        return connector.get_options_chain_snapshot(symbol)


    @st.cache_data(ttl=900, show_spinner=False)
    def get_correlation_matrix_cached(symbols_key: Tuple[str, ...]):
        return connector.get_correlation_matrix(list(symbols_key))


    @st.cache_data(ttl=900, show_spinner=False)
    def get_historical_metrics_cached(symbol: str):
        return connector.get_historical_metrics(symbol)


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
            help="Symbols used for tables, correlation, and comparison panels.",
        )
        show_3d_surface = st.checkbox("3D surface", value=True)
        show_correlations = st.checkbox("Realized correlation", value=True)
        show_chain = st.checkbox("Option chain", value=True)
        auto_refresh = st.checkbox("Auto refresh", value=False)
        refresh_interval = st.slider("Refresh interval seconds", 15, 180, 60, 15)
        st.markdown("### Data Filters")
        max_spread_pct = st.slider("Max spread percent", 0.05, 1.50, 0.75, 0.05)
        min_open_interest = st.number_input("Min open interest", min_value=0, value=0, step=10)
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
        help="Main symbol used for the volatility surface workspace.",
    )

    with st.spinner(f"Loading {surface_symbol} market data and surface..."):
        current_data = get_current_data_cached(surface_symbol)
        strikes, expiries, vol_surface = get_vol_surface_data_cached(surface_symbol)
        surface_meta = get_surface_metadata_cached(surface_symbol)
        stats = surface_stats(strikes, expiries, vol_surface, current_data["price"])

    surface_mode = surface_meta.get("surface_mode") or current_data.get("data_mode", "Unknown")
    source_label = surface_meta.get("surface_source") or current_data.get("price_source", "Unknown")

    st.markdown(
        f"""
    <div class="workstation-header">
        <div class="workstation-title">Options Volatility Surface Workstation</div>
        <div class="workstation-subtitle">
            {surface_symbol} | Spot {fmt_money(current_data.get("price"))} |
            Source {source_label} | Updated {datetime.now().strftime("%H:%M:%S")}
        </div>
        <div style="margin-top:0.55rem;">
            {status_pill("Surface", surface_mode)}
            {status_pill("Price", current_data.get("data_mode", "Unknown"))}
            {status_pill("IV", current_data.get("iv_source", "Unknown"))}
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
        ("Median Spread", fmt_pct(surface_meta.get("median_spread_pct"), 2), "option chain"),
    ]
    for col, (label, value, delta) in zip(kpi_cols, kpis):
        with col:
            st.metric(label, value, delta=delta if delta else None)

    st.markdown(
        f"""
    <div class="quality-row">
        <strong>Data quality:</strong>
        raw rows {fmt_int(surface_meta.get("raw_rows"))};
        valid rows {fmt_int(surface_meta.get("valid_rows"))};
        rejected rows {fmt_int(surface_meta.get("rejected_rows"))};
        cache age {fmt_int(surface_meta.get("cache_age_seconds"))}s;
        fallback reason {surface_meta.get("fallback_reason") or "none"}.
    </div>
    """,
        unsafe_allow_html=True,
    )

    market_rows = []
    for symbol in selected_symbols:
        data = get_current_data_cached(symbol)
        market_rows.append(
            {
                "Symbol": symbol,
                "Spot": data.get("price"),
                "30D IV": data.get("iv_30d"),
                "60D IV": data.get("iv_60d"),
                "90D IV": data.get("iv_90d"),
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
    market_df = pd.DataFrame(market_rows)

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
        st.dataframe(
            market_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Spot": st.column_config.NumberColumn(format="$%.2f"),
                "30D IV": st.column_config.NumberColumn(format="%.2f"),
                "60D IV": st.column_config.NumberColumn(format="%.2f"),
                "90D IV": st.column_config.NumberColumn(format="%.2f"),
                "Delta": st.column_config.NumberColumn(format="%.4f"),
                "Gamma": st.column_config.NumberColumn(format="%.4f"),
                "Theta/day": st.column_config.NumberColumn(format="%.4f"),
                "Vega/1%": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    with chain_tab:
        st.markdown('<div class="section-header">Option Chain Explorer</div>', unsafe_allow_html=True)
        chain_df, chain_meta = get_options_chain_cached(surface_symbol)
        if show_chain and not chain_df.empty:
            filtered = chain_df.copy()
            if "bidAskSpreadPct" in filtered:
                filtered = filtered[(filtered["bidAskSpreadPct"].isna()) | (filtered["bidAskSpreadPct"] <= max_spread_pct)]
            if "openInterest" in filtered:
                filtered = filtered[filtered["openInterest"].fillna(0) >= min_open_interest]

            display_cols = [
                "type",
                "expiration",
                "daysToExpiration",
                "strike",
                "moneyness",
                "bid",
                "ask",
                "mid",
                "last",
                "volume",
                "openInterest",
                "impliedVolatility",
                "bidAskSpreadPct",
            ]
            display_cols = [col for col in display_cols if col in filtered.columns]
            st.dataframe(
                filtered[display_cols],
                width="stretch",
                hide_index=True,
                column_config={
                    "expiration": st.column_config.DateColumn("Expiration"),
                    "strike": st.column_config.NumberColumn(format="$%.2f"),
                    "moneyness": st.column_config.NumberColumn(format="%.3f"),
                    "bid": st.column_config.NumberColumn(format="$%.2f"),
                    "ask": st.column_config.NumberColumn(format="$%.2f"),
                    "mid": st.column_config.NumberColumn(format="$%.2f"),
                    "last": st.column_config.NumberColumn(format="$%.2f"),
                    "impliedVolatility": st.column_config.NumberColumn("IV", format="%.2f"),
                    "bidAskSpreadPct": st.column_config.NumberColumn("Spread %", format="%.2f"),
                },
            )
            st.caption(
                f"Showing {len(filtered):,} of {len(chain_df):,} valid contracts. "
                f"Source: {chain_meta.get('source', 'unknown')}; mode: {chain_meta.get('mode', 'unknown')}."
            )
        else:
            st.info("No real option chain is available for this symbol. The surface may be using an explicit synthetic fallback.")

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

        hist_metrics = get_historical_metrics_cached(surface_symbol)
        if hist_metrics.get("available"):
            r20 = hist_metrics.get("realized_20d_latest")
            r60 = hist_metrics.get("realized_60d_latest")
            st.caption(
                f"Realized volatility from {hist_metrics.get('source')}: "
                f"20D {fmt_pct(r20)}, 60D {fmt_pct(r60)}."
            )
        else:
            st.caption(f"Realized volatility unavailable: {hist_metrics.get('reason', 'unknown reason')}.")

    with risk_tab:
        st.markdown('<div class="section-header">Portfolio And Cross-Asset Risk</div>', unsafe_allow_html=True)
        portfolio = connector.get_portfolio_metrics()
        if not portfolio.get("configured"):
            st.info(
                "No position book is configured yet. Random VaR, Sharpe, drawdown, and P&L have been removed from live mode. "
                "Use this panel for realized correlations until a real position import is added."
            )

        if show_correlations and len(selected_symbols) > 1:
            corr = get_correlation_matrix_cached(tuple(sorted(selected_symbols)))
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
                st.warning("Not enough historical data was available to compute realized correlations.")
        else:
            st.caption("Select at least two symbols and enable realized correlation to use this panel.")

    with diagnostics_tab:
        st.markdown('<div class="section-header">Diagnostics And Data Provenance</div>', unsafe_allow_html=True)
        health = connector.get_system_health()
        col1, col2 = st.columns(2)
        with col1:
            st.json(health.get("overall", {}))
        with col2:
            st.json(health.get("data_contract", {}))

        st.markdown("#### Latest Surface Metadata")
        st.json({k: str(v) if isinstance(v, datetime) else v for k, v in surface_meta.items()})
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
