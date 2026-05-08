from __future__ import annotations

import pandas as pd
import pytest

from src.quant.advanced_features import (
    broker_integration_abstraction,
    build_option_strategy,
    compare_saved_snapshots,
    cross_sectional_vol_map,
    earnings_vol_event_engine,
    estimate_transaction_costs,
    evaluate_surface_alerts,
    export_analysis_notebook,
    list_surface_workspaces,
    optimize_portfolio_hedges,
    paper_trading_simulator,
    parse_portfolio_positions,
    portfolio_risk_summary,
    relative_value_dashboard,
    run_signal_backtest,
    load_surface_workspace,
    save_surface_workspace,
    strategy_scenario_engine,
    surface_iv_for_contract,
    watchlist_presets,
)


def _strategy_chain() -> pd.DataFrame:
    rows = []
    for expiry, dte, base_iv in (
        (pd.Timestamp("2026-06-19"), 30, 0.24),
        (pd.Timestamp("2026-07-17"), 58, 0.27),
    ):
        for strike in (90.0, 100.0, 110.0, 120.0):
            for option_type in ("call", "put"):
                rows.append(
                    {
                        "contractSymbol": f"AAPL{expiry:%y%m%d}{option_type[0].upper()}{int(strike * 1000):08d}",
                        "type": option_type,
                        "expiration": expiry,
                        "daysToExpiration": dte,
                        "strike": strike,
                        "computedIV": base_iv + abs(strike / 100.0 - 1.0) * 0.04,
                        "selectedMarketPrice": 5.0,
                        "riskFreeRate": 0.05,
                        "effectiveDividendYield": 0.01,
                    }
                )
    return pd.DataFrame(rows)


def test_relative_value_dashboard_returns_pair_spreads_and_normalized_overlay():
    result = relative_value_dashboard(
        {
            "symbol": "AAPL",
            "iv_30d": 0.30,
            "iv_90d": 0.36,
            "front_risk_reversal_25d": -0.04,
            "realized_20d_latest": 0.22,
            "iv_rank": 0.80,
        },
        {
            "symbol": "MSFT",
            "iv_30d": 0.24,
            "iv_90d": 0.27,
            "front_risk_reversal_25d": -0.02,
            "realized_20d_latest": 0.20,
            "iv_rank": 0.55,
        },
    )

    assert result["available"] is True
    assert result["spreads"]["atm_iv_spread"] == pytest.approx(0.06)
    assert result["spreads"]["skew_spread"] == pytest.approx(-0.02)
    assert result["spreads"]["term_slope_spread"] == pytest.approx(0.03)
    assert result["spreads"]["realized_spread"] == pytest.approx(0.04)
    assert {row["metric"] for row in result["normalized_overlays"]} >= {"ATM IV", "IV - Realized"}


def test_cross_sectional_vol_map_scores_and_sorts_universe():
    result = cross_sectional_vol_map(
        [
            {"symbol": "AAPL", "iv_rank": 0.85, "iv_percentile": 0.80, "skew_25d": -0.07, "term_slope": 0.05, "iv_realized_spread": 0.11},
            {"symbol": "MSFT", "iv_rank": 0.35, "iv_percentile": 0.40, "skew_25d": -0.01, "term_slope": 0.01, "iv_realized_spread": 0.02},
            {"symbol": "SPY", "iv_rank": 0.55, "iv_percentile": 0.50, "skew_25d": -0.03, "term_slope": 0.02, "iv_realized_spread": 0.04},
        ]
    )

    assert result["available"] is True
    assert result["symbol_count"] == 3
    assert result["opportunities"][0]["symbol"] == "AAPL"
    assert result["opportunities"][0]["rank"] == 1
    assert result["opportunities"][0]["opportunity_score"] > result["opportunities"][-1]["opportunity_score"]


def test_earnings_vol_event_engine_builds_event_card_with_crush():
    chain = _strategy_chain()
    events = [
        {
            "symbol": "AAPL",
            "event_type": "earnings",
            "event_date": "2026-06-15",
            "description": "AAPL earnings",
            "source": "fixture",
        }
    ]

    result = earnings_vol_event_engine(
        "AAPL",
        chain,
        100.0,
        events,
        historical_abs_moves=[0.035, 0.05, 0.04],
    )

    card = result["event_card"]
    assert result["available"] is True
    assert card["event_date"] == "2026-06-15"
    assert card["expiration"] == "2026-06-19"
    assert card["implied_move_pct"] > 0.0
    assert card["historical_avg_abs_move_pct"] == pytest.approx((0.035 + 0.05 + 0.04) / 3)
    assert card["post_event_crush"] is not None


def test_surface_iv_interpolation_and_strategy_pricing_use_fitted_surface():
    strikes = [90.0, 100.0, 110.0]
    expiries = [30.0, 60.0]
    surface = [[0.30, 0.24, 0.28], [0.34, 0.27, 0.31]]

    assert surface_iv_for_contract(strikes, expiries, surface, 105.0, 45.0) == pytest.approx(0.275)

    strategy = build_option_strategy(
        _strategy_chain(),
        100.0,
        "straddle",
        strike_grid=strikes,
        expiry_grid=expiries,
        surface=surface,
    )

    assert strategy["available"] is True
    assert strategy["strategy_type"] == "straddle"
    assert strategy["leg_count"] == 2
    assert strategy["surface_priced_legs"] == 2
    assert strategy["net_debit"] > 0.0
    assert strategy["greeks"]["vega"] > 0.0
    assert len(strategy["breakevens"]) == 2
    assert strategy["max_profit_100x"] > strategy["max_loss_100x"]


def test_strategy_scenario_engine_returns_spot_vol_and_time_axes():
    strategy = build_option_strategy(_strategy_chain(), 100.0, "straddle")

    scenarios = strategy_scenario_engine(
        strategy,
        100.0,
        spot_shifts=[-0.05, 0.0, 0.05],
        time_pass_days=[0.0, 7.0],
        vol_shifts=[-0.02, 0.0, 0.02],
        skew_shifts=[0.0],
    )

    assert scenarios["available"] is True
    assert scenarios["axes"]["spot_shifts"] == [-0.05, 0.0, 0.05]
    assert len(scenarios["points"]) == 18
    assert len(scenarios["spot_vol_heatmap"]) == 9
    assert any(row["pnl_100x"] > 0.0 for row in scenarios["spot_vol_heatmap"])


def test_portfolio_import_risk_and_optimization_are_deterministic():
    parsed = parse_portfolio_positions(
        "symbol,expiry,strike,type,quantity,cost\n"
        "AAPL,2026-06-19,100,call,2,4.50\n"
        "AAPL,2026-06-19,90,put,-1,2.00\n"
    )
    market = {"AAPL": {"spot": 100.0, "chain": _strategy_chain()}}

    portfolio = portfolio_risk_summary(parsed["positions"], market)
    optimization = optimize_portfolio_hedges(portfolio, objective="delta-neutral")
    max_loss = optimize_portfolio_hedges(portfolio, objective="max loss constraint")

    assert parsed["available"] is True
    assert portfolio["available"] is True
    assert portfolio["totals"]["market_value_100x"] > 0.0
    assert portfolio["totals"]["delta_100x"] != 0.0
    assert len(portfolio["scenario_pnl"]) == 9
    assert optimization["available"] is True
    assert {"contract", "size", "estimated_cost", "trade_offs"}.issubset(optimization["suggestions"][0])
    assert max_loss["available"] is True
    assert max_loss["objective"] == "max loss constraint"


def test_alerts_log_locally_and_watchlist_presets_include_events(tmp_path):
    log_path = tmp_path / "alerts.jsonl"

    alerts = evaluate_surface_alerts(
        "AAPL",
        {
            "iv_rank": 0.91,
            "front_risk_reversal_25d": -0.08,
            "svi_rmse": 0.05,
            "rich_cheap_scanner": {"candidates": [{"residual": -0.14}]},
        },
        {"data_delay_minutes": 45},
        log_path=log_path,
        timestamp=pd.Timestamp("2026-05-07T12:00:00").to_pydatetime(),
    )
    presets = watchlist_presets(
        [{"symbol": "AAPL", "event_type": "earnings", "event_date": "2026-05-09"}],
        as_of=pd.Timestamp("2026-05-07").date(),
    )

    assert alerts["alert_count"] == 5
    assert log_path.read_text(encoding="utf-8").count("\n") == 5
    assert presets["Mega-cap tech"][:2] == ["AAPL", "MSFT"]
    assert presets["Earnings this week"] == ["AAPL"]


def test_saved_workspaces_round_trip_selected_state_and_provenance(tmp_path):
    saved = save_surface_workspace(
        {
            "name": "Morning Surface",
            "selected_symbols": ["aapl", "msft"],
            "filters": {"min_open_interest": 100, "moneyness": [0.8, 1.2]},
            "model_settings": {"pricing_model": "bsm_dividends", "smoothing": "svi"},
            "chart_layout": {"surface_axis": "log_moneyness", "show_3d": True},
            "provenance": {"source": "unit-test", "data_mode": "Synthetic"},
        },
        tmp_path,
        timestamp=pd.Timestamp("2026-05-08T09:30:00").to_pydatetime(),
    )

    loaded = load_surface_workspace(saved["path"])
    listed = list_surface_workspaces(tmp_path)

    assert saved["available"] is True
    assert loaded["workspace"]["selected_symbols"] == ["AAPL", "MSFT"]
    assert loaded["workspace"]["filters"]["min_open_interest"] == 100
    assert loaded["workspace"]["model_settings"]["pricing_model"] == "bsm_dividends"
    assert loaded["workspace"]["chart_layout"]["surface_axis"] == "log_moneyness"
    assert loaded["workspace"]["provenance"]["source"] == "unit-test"
    assert listed[0]["name"] == "Morning Surface"


def test_snapshot_comparison_reports_surface_skew_term_and_scanner_deltas():
    left = _strategy_chain().assign(
        selectedMarketPrice=lambda frame: frame["selectedMarketPrice"] + frame["strike"] * 0.01,
        residual=[-0.02, 0.01, -0.03, 0.02, -0.04, 0.03, -0.05, 0.04] * 2,
    )
    right = left.copy()
    right["computedIV"] = right["computedIV"] + 0.015
    right["selectedMarketPrice"] = right["selectedMarketPrice"] + 0.25
    right["residual"] = right["residual"] + 0.02

    comparison = compare_saved_snapshots(
        {"symbol": "AAPL", "timestamp": "2026-05-08T09:30:00", "options": left},
        {"symbol": "AAPL", "timestamp": "2026-05-08T10:30:00", "options": right},
    )

    assert comparison["available"] is True
    assert comparison["surface_deltas"]["matched_points"] == len(left)
    assert comparison["surface_deltas"]["mean_iv_delta"] == pytest.approx(0.015)
    assert comparison["surface_deltas"]["mean_price_delta"] == pytest.approx(0.25)
    assert comparison["skew_deltas"]
    assert comparison["term_deltas"][0]["median_iv_delta"] == pytest.approx(0.015)
    assert comparison["scanner_deltas"]


def test_transaction_cost_model_and_backtest_use_explicit_costs():
    costs = estimate_transaction_costs(
        [
            {"symbol": "AAPL", "action": "buy", "quantity": 2, "price": 4.0, "bid": 3.9, "ask": 4.1},
            {"symbol": "AAPL", "action": "assignment", "quantity": 1, "price": 0.0},
        ],
        commission_per_contract=0.5,
        slippage_bps=5.0,
        assignment_fee=4.0,
    )
    backtest = run_signal_backtest(
        [
            {"date": "2026-05-01", "symbol": "AAPL", "close": 100.0, "iv_rank": 0.20, "bid": 99.9, "ask": 100.1},
            {"date": "2026-05-02", "symbol": "AAPL", "close": 102.0, "iv_rank": 0.85, "bid": 101.9, "ask": 102.1},
            {"date": "2026-05-03", "symbol": "AAPL", "close": 99.0, "residual": 0.10, "bid": 98.9, "ask": 99.1},
            {"date": "2026-05-04", "symbol": "AAPL", "close": 98.0, "skew_25d": -0.08, "bid": 97.9, "ask": 98.1},
        ],
        initial_cash=100000.0,
        notional=10000.0,
        cost_config={"commission_per_contract": 0.0, "slippage_bps": 0.0},
    )

    assert costs["total_cost"] == pytest.approx(25.9)
    assert costs["assignment_exercise_fees"] == pytest.approx(4.0)
    assert backtest["available"] is True
    assert {"return", "max_drawdown", "hit_rate", "sharpe", "turnover", "transaction_costs"}.issubset(backtest)
    assert backtest["turnover"] > 0.0
    assert backtest["transaction_costs"] > 0.0
    assert any(row["signal"] == "high IV rank" for row in backtest["rows"])


def test_paper_trading_marks_positions_without_broker_and_broker_is_read_only():
    paper = paper_trading_simulator(
        [{"symbol": "AAPL", "side": "buy", "quantity": 2, "price": 4.0, "bid": 3.9, "ask": 4.1}],
        {"AAPL": 4.8},
        starting_cash=10000.0,
        cost_config={"commission_per_contract": 0.0, "slippage_bps": 0.0},
        timestamp=pd.Timestamp("2026-05-08T09:35:00").to_pydatetime(),
    )
    broker = broker_integration_abstraction(paper["positions"], account={"id": "paper-readonly"})

    assert paper["available"] is True
    assert paper["broker_required"] is False
    assert paper["positions"][0]["unrealized_pnl"] == pytest.approx(160.0)
    assert paper["order_log"][0]["mode"] == "paper"
    assert broker["capabilities"]["read_positions"] is True
    assert broker["capabilities"]["place_orders"] is False
    assert broker["trade_submission"]["enabled"] is False


def test_notebook_export_writes_reproducible_payload_with_timestamp(tmp_path):
    output = tmp_path / "surface_report.ipynb"
    result = export_analysis_notebook(
        {
            "symbol": "AAPL",
            "data_timestamp": "2026-05-08T09:30:00",
            "model_assumptions": "BSM with dividends, SVI smoothing",
            "surface_deltas": {"mean_iv_delta": 0.01},
        },
        output,
        title="AAPL Surface Export",
        timestamp=pd.Timestamp("2026-05-08T09:45:00").to_pydatetime(),
    )

    notebook = output.read_text(encoding="utf-8")
    assert result["available"] is True
    assert result["cell_count"] == 2
    assert '"nbformat": 4' in notebook
    assert "2026-05-08T09:30:00" in notebook
    assert "BSM with dividends, SVI smoothing" in notebook
