from __future__ import annotations

import pandas as pd
import pytest

from src.quant.advanced_features import (
    build_option_strategy,
    cross_sectional_vol_map,
    earnings_vol_event_engine,
    evaluate_surface_alerts,
    optimize_portfolio_hedges,
    parse_portfolio_positions,
    portfolio_risk_summary,
    relative_value_dashboard,
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
