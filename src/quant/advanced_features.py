"""Phase 4 relative-value, workflow, event, and strategy analytics."""

from __future__ import annotations

import json
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pricing.black_scholes import BlackScholesModel, OptionGreeks
from src.quant.expected_move import expected_moves_by_expiry


CONTRACT_MULTIPLIER = 100
WORKSPACE_SCHEMA_VERSION = 1
NOTEBOOK_NBFORMAT = 4


def relative_value_dashboard(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    """Compare two underlyings across normalized volatility metrics."""
    left_profile = _vol_profile(left)
    right_profile = _vol_profile(right)
    if not left_profile or not right_profile:
        return _unavailable("Need two symbols with representative volatility metrics")

    metrics = {
        "atm_iv_spread": _spread(left_profile.get("atm_iv"), right_profile.get("atm_iv")),
        "skew_spread": _spread(left_profile.get("skew_25d"), right_profile.get("skew_25d")),
        "term_slope_spread": _spread(left_profile.get("term_slope"), right_profile.get("term_slope")),
        "realized_spread": _spread(
            left_profile.get("iv_realized_spread"),
            right_profile.get("iv_realized_spread"),
        ),
    }
    overlays = _normalized_overlay_records(left_profile, right_profile)
    return {
        "available": bool(overlays),
        "source": "symbol_profiles",
        "left_symbol": left_profile["symbol"],
        "right_symbol": right_profile["symbol"],
        "profiles": [left_profile, right_profile],
        "spreads": metrics,
        "normalized_overlays": overlays,
    }


def cross_sectional_vol_map(rows: list[dict[str, Any]], *, limit: int | None = None) -> dict[str, Any]:
    """Rank a symbol universe by IV rank, skew, term slope, and implied-realized spread."""
    profiles = [_vol_profile(row) for row in rows]
    profiles = [profile for profile in profiles if profile]
    if not profiles:
        return _unavailable("No symbols have usable cross-sectional volatility metrics")

    frame = pd.DataFrame(profiles)
    score_parts = []
    for column in ("iv_rank", "iv_percentile", "skew_25d", "term_slope", "iv_realized_spread"):
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        z = _zscore(values)
        if column in {"skew_25d", "term_slope"}:
            z = z.abs()
        score_parts.append(z.fillna(0.0))
    if score_parts:
        frame["opportunity_score"] = sum(score_parts) / len(score_parts)
    else:
        frame["opportunity_score"] = 0.0

    frame = frame.sort_values(["opportunity_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    if limit is not None:
        frame = frame.head(limit)

    return {
        "available": True,
        "source": "symbol_profiles",
        "symbol_count": int(len(profiles)),
        "metrics": [
            "iv_rank",
            "iv_percentile",
            "skew_25d",
            "term_slope",
            "iv_realized_spread",
        ],
        "opportunities": frame.replace({np.nan: None}).to_dict("records"),
    }


def earnings_vol_event_engine(
    symbol: str,
    chain: pd.DataFrame,
    spot: float,
    events: Any,
    *,
    historical_abs_moves: list[float] | tuple[float, ...] | None = None,
    iv_column: str = "computedIV",
    price_column: str = "selectedMarketPrice",
) -> dict[str, Any]:
    """Estimate the next earnings move and post-event IV crush."""
    earnings = _earnings_events(events)
    if not earnings:
        return _unavailable("No upcoming earnings event found")
    if chain.empty:
        return _unavailable("Option chain is empty")

    event = earnings[0]
    event_date = pd.to_datetime(event.get("event_date"), errors="coerce")
    if pd.isna(event_date):
        return _unavailable("Earnings event date is invalid")

    moves = expected_moves_by_expiry(chain, spot, iv_column=iv_column, price_column=price_column)
    if moves.empty:
        return _unavailable("Expected-move inputs are unavailable")

    moves = moves.copy()
    moves["expiration_dt"] = pd.to_datetime(moves["expiration"], errors="coerce")
    after_event = moves[moves["expiration_dt"] >= event_date.normalize()].sort_values("expiration_dt")
    selected = after_event.iloc[0] if not after_event.empty else moves.sort_values("expiration_dt").iloc[0]
    atm_term = _atm_iv_by_expiry(chain, spot, iv_column)
    crush = _post_event_crush(atm_term, pd.to_datetime(selected["expiration"]))

    historical_clean = [float(value) for value in (historical_abs_moves or []) if _is_finite(value)]
    historical_avg = float(np.mean(np.abs(historical_clean))) if historical_clean else None
    implied_pct = _finite_or_none(selected.get("expected_move_pct"))
    return {
        "available": True,
        "source": "event_calendar_plus_option_chain",
        "symbol": symbol.upper(),
        "event_card": {
            "event_date": event_date.date().isoformat(),
            "description": event.get("description") or "Earnings",
            "source": event.get("source") or "event_calendar",
            "expiration": str(selected.get("expiration")),
            "dte": _finite_or_none(selected.get("dte")),
            "implied_move": _finite_or_none(selected.get("expected_move")),
            "implied_move_pct": implied_pct,
            "method": selected.get("method"),
            "historical_avg_abs_move_pct": historical_avg,
            "implied_vs_historical_spread": (
                None if implied_pct is None or historical_avg is None else implied_pct - historical_avg
            ),
            "post_event_crush": crush,
        },
        "historical_observations": len(historical_clean),
        "atm_term": atm_term,
    }


def surface_iv_for_contract(
    strike_grid: Any,
    expiry_grid: Any,
    surface: Any,
    strike: float,
    dte: float,
) -> float | None:
    """Interpolate fitted surface IV for one strike and expiry."""
    strikes, expiries, vols = _surface_axes(strike_grid, expiry_grid, surface)
    if strikes.size == 0 or expiries.size == 0 or vols.size == 0:
        return None
    if not (_is_finite(strike) and _is_finite(dte)):
        return None

    row_values = []
    for row in vols:
        valid = np.isfinite(row) & np.isfinite(strikes)
        if valid.sum() == 0:
            row_values.append(np.nan)
        elif valid.sum() == 1:
            row_values.append(float(row[valid][0]))
        else:
            row_values.append(float(np.interp(float(strike), strikes[valid], row[valid])))
    row_values_arr = np.asarray(row_values, dtype=float)
    valid_expiries = np.isfinite(row_values_arr) & np.isfinite(expiries)
    if valid_expiries.sum() == 0:
        return None
    if valid_expiries.sum() == 1:
        return float(row_values_arr[valid_expiries][0])
    return float(np.interp(float(dte), expiries[valid_expiries], row_values_arr[valid_expiries]))


def build_option_strategy(
    chain: pd.DataFrame,
    spot: float,
    strategy_type: str,
    *,
    strike_grid: Any | None = None,
    expiry_grid: Any | None = None,
    surface: Any | None = None,
) -> dict[str, Any]:
    """Create a template strategy and price legs with fitted surface IV when available."""
    prepared = _prepared_chain(chain)
    if prepared.empty:
        return _unavailable("No option rows are available for strategy construction")
    legs = _strategy_template(prepared, spot, strategy_type)
    if not legs:
        return _unavailable(f"Could not construct {strategy_type} from available strikes and expiries")
    return price_option_strategy(
        prepared,
        spot,
        legs,
        strategy_type=strategy_type,
        strike_grid=strike_grid,
        expiry_grid=expiry_grid,
        surface=surface,
    )


def price_option_strategy(
    chain: pd.DataFrame,
    spot: float,
    legs: list[dict[str, Any]],
    *,
    strategy_type: str = "custom",
    strike_grid: Any | None = None,
    expiry_grid: Any | None = None,
    surface: Any | None = None,
) -> dict[str, Any]:
    """Price explicit strategy legs and summarize payoff, Greeks, and breakevens."""
    prepared = _prepared_chain(chain)
    priced_legs = []
    for leg in legs:
        row = _match_leg(prepared, leg)
        if row is None:
            continue
        priced = _price_leg(row, spot, leg, strike_grid, expiry_grid, surface)
        if priced:
            priced_legs.append(priced)
    if not priced_legs:
        return _unavailable("No strategy legs could be matched to option rows")

    net_debit = float(sum(item["quantity"] * item["model_price"] for item in priced_legs))
    payoff = _strategy_payoff(priced_legs, spot, net_debit)
    return {
        "available": True,
        "source": "option_chain_plus_fitted_surface",
        "strategy_type": _strategy_key(strategy_type),
        "pricing_model": "Black-Scholes with fitted surface IV where available",
        "surface_priced_legs": int(sum(1 for item in priced_legs if item["surface_iv"] is not None)),
        "leg_count": len(priced_legs),
        "net_debit": net_debit,
        "net_debit_100x": net_debit * CONTRACT_MULTIPLIER,
        "legs": priced_legs,
        "greeks": {
            greek: float(sum(item["quantity"] * item[greek] for item in priced_legs))
            for greek in ("delta", "gamma", "theta", "vega")
        },
        **payoff,
    }


def strategy_scenario_engine(
    strategy: dict[str, Any],
    spot: float,
    *,
    spot_shifts: list[float] | tuple[float, ...] | None = None,
    time_pass_days: list[float] | tuple[float, ...] | None = None,
    vol_shifts: list[float] | tuple[float, ...] | None = None,
    skew_shifts: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Reprice a strategy across spot, time, parallel-vol, and skew shocks."""
    if not strategy.get("available"):
        return _unavailable(strategy.get("reason") or "Strategy is unavailable")
    legs = [dict(item) for item in strategy.get("legs") or []]
    if not legs:
        return _unavailable("Strategy has no priced legs")
    if not _is_finite(spot) or spot <= 0.0:
        return _unavailable("Scenario spot must be positive")

    spot_axis = tuple(float(v) for v in (spot_shifts or (-0.10, -0.05, 0.0, 0.05, 0.10)))
    time_axis = tuple(float(v) for v in (time_pass_days or (0.0, 7.0, 14.0, 30.0)))
    vol_axis = tuple(float(v) for v in (vol_shifts or (-0.05, 0.0, 0.05)))
    skew_axis = tuple(float(v) for v in (skew_shifts or (-0.03, 0.0, 0.03)))
    base_value = _strategy_value(legs, float(spot), time_pass=0.0, vol_shift=0.0, skew_shift=0.0)

    points = []
    for spot_shift in spot_axis:
        shocked_spot = float(spot) * (1.0 + spot_shift)
        if shocked_spot <= 0.0:
            continue
        for time_pass in time_axis:
            for vol_shift in vol_axis:
                for skew_shift in skew_axis:
                    value = _strategy_value(
                        legs,
                        shocked_spot,
                        time_pass=time_pass,
                        vol_shift=vol_shift,
                        skew_shift=skew_shift,
                        base_spot=float(spot),
                    )
                    pnl = value - base_value
                    points.append(
                        {
                            "spot_shift": spot_shift,
                            "time_pass_days": time_pass,
                            "vol_shift": vol_shift,
                            "skew_shift": skew_shift,
                            "shocked_spot": shocked_spot,
                            "value": value,
                            "pnl": pnl,
                            "pnl_100x": pnl * CONTRACT_MULTIPLIER,
                        }
                    )

    neutral_skew = min(skew_axis, key=abs)
    neutral_time = min(time_axis, key=abs)
    neutral_vol = min(vol_axis, key=abs)
    return {
        "available": bool(points),
        "source": "black_scholes_strategy_repricing",
        "base_value": base_value,
        "base_value_100x": base_value * CONTRACT_MULTIPLIER,
        "axes": {
            "spot_shifts": list(spot_axis),
            "time_pass_days": list(time_axis),
            "vol_shifts": list(vol_axis),
            "skew_shifts": list(skew_axis),
        },
        "points": points,
        "spot_vol_heatmap": [
            row
            for row in points
            if row["time_pass_days"] == neutral_time and row["skew_shift"] == neutral_skew
        ],
        "spot_time_heatmap": [
            row
            for row in points
            if row["vol_shift"] == neutral_vol and row["skew_shift"] == neutral_skew
        ],
    }


def parse_portfolio_positions(csv_input: Any) -> dict[str, Any]:
    """Parse a CSV option-position upload with deterministic validation."""
    if csv_input is None:
        return _unavailable("No CSV content provided")
    if isinstance(csv_input, pd.DataFrame):
        frame = csv_input.copy()
    else:
        text = csv_input.decode("utf-8-sig") if isinstance(csv_input, bytes) else str(csv_input)
        frame = pd.read_csv(StringIO(text))

    rename = {str(column).strip().lower(): column for column in frame.columns}
    required = ("symbol", "expiry", "strike", "type", "quantity", "cost")
    missing = [name for name in required if name not in rename]
    if missing:
        return _unavailable(f"Missing required columns: {', '.join(missing)}")

    out = pd.DataFrame({name: frame[rename[name]] for name in required})
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.date
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["type"] = out["type"].astype(str).str.strip().str.lower()
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["cost"] = pd.to_numeric(out["cost"], errors="coerce")
    valid_type = out["type"].isin({"call", "put"})
    valid = (
        out["symbol"].ne("")
        & out["expiry"].notna()
        & out["strike"].gt(0.0)
        & valid_type
        & out["quantity"].notna()
        & out["cost"].notna()
    )
    errors = []
    for index, row in out[~valid].iterrows():
        reasons = []
        if not row["symbol"]:
            reasons.append("symbol")
        if pd.isna(row["expiry"]):
            reasons.append("expiry")
        if not _is_finite(row["strike"]) or row["strike"] <= 0.0:
            reasons.append("strike")
        if row["type"] not in {"call", "put"}:
            reasons.append("type")
        if not _is_finite(row["quantity"]):
            reasons.append("quantity")
        if not _is_finite(row["cost"]):
            reasons.append("cost")
        errors.append({"row": int(index) + 2, "fields": reasons})
    clean = out[valid].reset_index(drop=True)
    return {
        "available": not clean.empty,
        "source": "csv_upload",
        "position_count": int(len(clean)),
        "errors": errors,
        "positions": clean.to_dict("records"),
    }


def portfolio_risk_summary(
    positions: list[dict[str, Any]],
    market_data: dict[str, dict[str, Any]],
    *,
    surface_grids: dict[str, tuple[Any, Any, Any]] | None = None,
    scenario_spot_shifts: list[float] | tuple[float, ...] | None = None,
    scenario_vol_shifts: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Price uploaded positions, aggregate Greeks, and run portfolio scenarios."""
    surface_grids = surface_grids or {}
    priced = []
    unmatched = []
    for position in positions:
        symbol = str(position.get("symbol", "")).upper()
        payload = market_data.get(symbol) or {}
        chain = payload.get("chain")
        spot = _finite_or_none(payload.get("spot"))
        if not isinstance(chain, pd.DataFrame) or chain.empty or spot is None:
            unmatched.append({**position, "reason": "missing market data"})
            continue
        prepared = _prepared_chain(chain)
        row = _match_leg(
            prepared,
            {
                "type": position.get("type"),
                "expiration": position.get("expiry") or position.get("expiration"),
                "strike": position.get("strike"),
                "quantity": position.get("quantity", 1.0),
            },
        )
        if row is None:
            unmatched.append({**position, "reason": "contract not found"})
            continue
        strike_grid, expiry_grid, surface = surface_grids.get(symbol, (None, None, None))
        priced_leg = _price_leg(row, spot, position, strike_grid, expiry_grid, surface)
        if not priced_leg:
            unmatched.append({**position, "reason": "contract could not be priced"})
            continue
        quantity = float(position.get("quantity", 0.0))
        cost = float(position.get("cost", 0.0))
        priced.append(
            {
                **priced_leg,
                "symbol": symbol,
                "spot": spot,
                "quantity": quantity,
                "cost": cost,
                "market_value_100x": quantity * priced_leg["model_price"] * CONTRACT_MULTIPLIER,
                "cost_basis_100x": quantity * cost * CONTRACT_MULTIPLIER,
                "unrealized_pnl_100x": quantity * (priced_leg["model_price"] - cost) * CONTRACT_MULTIPLIER,
            }
        )

    if not priced:
        return {
            "configured": bool(positions),
            "available": False,
            "reason": "No uploaded positions could be matched to market data",
            "positions": [],
            "unmatched": unmatched,
        }

    totals = _aggregate_position_greeks(priced)
    scenarios = _portfolio_scenarios(
        priced,
        scenario_spot_shifts or (-0.05, 0.0, 0.05),
        scenario_vol_shifts or (-0.03, 0.0, 0.03),
    )
    return {
        "configured": True,
        "available": True,
        "source": "csv_upload_plus_option_chain",
        "position_count": len(priced),
        "unmatched_count": len(unmatched),
        "positions": priced,
        "unmatched": unmatched,
        "totals": totals,
        "scenario_pnl": scenarios,
        "total_value": totals["market_value_100x"],
        "daily_pnl": totals["unrealized_pnl_100x"],
        "var_95": abs(min((row["pnl_100x"] for row in scenarios), default=0.0)),
        "sharpe_ratio": None,
        "max_drawdown": None,
        "volatility": None,
    }


def optimize_portfolio_hedges(
    portfolio: dict[str, Any],
    *,
    objective: str = "delta-neutral",
    theta_target: float = 0.0,
    max_contracts: int = 10,
) -> dict[str, Any]:
    """Suggest simple hedge trades using currently priced portfolio contracts."""
    if not portfolio.get("available"):
        return _unavailable(portfolio.get("reason") or "Portfolio is unavailable")
    positions = portfolio.get("positions") or []
    totals = portfolio.get("totals") or {}
    objective_key = str(objective or "delta-neutral").strip().lower()
    if objective_key == "max loss constraint":
        worst = min((float(row.get("pnl_100x") or 0.0) for row in portfolio.get("scenario_pnl") or []), default=0.0)
        candidates = []
        for contract in positions:
            gamma = abs(float(contract.get("gamma") or 0.0)) * CONTRACT_MULTIPLIER
            vega = max(float(contract.get("vega") or 0.0), 0.0) * CONTRACT_MULTIPLIER
            protection_score = gamma + vega
            size = 1 if float(contract.get("quantity") or 0.0) <= 0.0 else -1
            estimated_cost = size * float(contract.get("model_price") or 0.0) * CONTRACT_MULTIPLIER
            candidates.append(
                {
                    "contract": contract.get("contract"),
                    "symbol": contract.get("symbol"),
                    "type": contract.get("type"),
                    "expiration": contract.get("expiration"),
                    "strike": contract.get("strike"),
                    "size": size,
                    "estimated_cost": estimated_cost,
                    "objective": objective_key,
                    "current_exposure": worst,
                    "post_trade_exposure": worst - abs(estimated_cost),
                    "target_exposure": 0.0,
                    "residual_exposure": abs(worst - abs(estimated_cost)) / max(protection_score, 1e-9),
                    "trade_offs": "Ranks listed contracts by convexity and vega protection; premium, liquidity, and basis risk remain explicit trade-offs.",
                }
            )
        candidates = sorted(candidates, key=lambda item: (item["residual_exposure"], abs(item["estimated_cost"])))
        return {
            "available": bool(candidates),
            "source": "deterministic_max_loss_hedge_scan",
            "objective": objective_key,
            "suggestions": candidates[:5],
        }

    target_map = {
        "delta-neutral": ("delta_100x", 0.0),
        "vega-neutral": ("vega_100x", 0.0),
        "theta target": ("theta_100x", float(theta_target)),
    }
    metric, target = target_map.get(objective_key, target_map["delta-neutral"])
    current = float(totals.get(metric) or 0.0)
    candidates = []
    for contract in positions:
        exposure = float(contract.get(metric.replace("_100x", "")) or 0.0) * CONTRACT_MULTIPLIER
        if abs(exposure) <= 1e-9:
            continue
        raw_size = (target - current) / exposure
        size = int(np.clip(np.round(raw_size), -max_contracts, max_contracts))
        if size == 0:
            size = 1 if raw_size > 0 else -1
        residual = current + size * exposure - target
        estimated_cost = size * float(contract.get("model_price") or 0.0) * CONTRACT_MULTIPLIER
        candidates.append(
            {
                "contract": contract.get("contract"),
                "symbol": contract.get("symbol"),
                "type": contract.get("type"),
                "expiration": contract.get("expiration"),
                "strike": contract.get("strike"),
                "size": size,
                "estimated_cost": estimated_cost,
                "objective": objective_key,
                "current_exposure": current,
                "post_trade_exposure": current + size * exposure,
                "target_exposure": target,
                "residual_exposure": residual,
                "trade_offs": _hedge_tradeoffs(contract, metric),
            }
        )
    candidates = sorted(candidates, key=lambda item: (abs(item["residual_exposure"]), abs(item["estimated_cost"])))
    return {
        "available": bool(candidates),
        "source": "deterministic_single_contract_hedge_scan",
        "objective": objective_key,
        "suggestions": candidates[:5],
    }


def evaluate_surface_alerts(
    symbol: str,
    metadata: dict[str, Any],
    current: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
    log_path: str | Path | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate configurable local alerts and optionally append JSONL records."""
    cfg = {
        "iv_rank_threshold": 0.80,
        "skew_steepening_threshold": 0.05,
        "surface_fit_error_threshold": 0.03,
        "data_stale_minutes": 30,
        "rich_cheap_residual_threshold": 0.10,
    }
    cfg.update(config or {})
    current = current or {}
    alerts = []
    key = symbol.upper()
    ts = timestamp or datetime.utcnow()

    def add(kind: str, severity: str, message: str, value: Any, threshold: Any) -> None:
        alerts.append(
            {
                "timestamp": ts.isoformat(),
                "symbol": key,
                "alert_type": kind,
                "severity": severity,
                "message": message,
                "value": _finite_or_none(value),
                "threshold": threshold,
            }
        )

    iv_rank = _finite_or_none(metadata.get("iv_rank"))
    if iv_rank is not None and iv_rank >= float(cfg["iv_rank_threshold"]):
        add("iv_rank_threshold", "warning", "IV rank is above configured threshold", iv_rank, cfg["iv_rank_threshold"])

    skew = _finite_or_none(metadata.get("front_risk_reversal_25d"))
    if skew is not None and abs(skew) >= float(cfg["skew_steepening_threshold"]):
        add("skew_steepening", "warning", "Front 25-delta skew is steepening", skew, cfg["skew_steepening_threshold"])

    fit_error = _first_finite(metadata, "svi_rmse", "ssvi_rmse", "surface_fit_rmse")
    if fit_error is not None and fit_error >= float(cfg["surface_fit_error_threshold"]):
        add("surface_fit_error", "critical", "Surface fit error exceeds configured tolerance", fit_error, cfg["surface_fit_error_threshold"])

    delay = _finite_or_none(current.get("data_delay_minutes") or metadata.get("data_delay_minutes"))
    if delay is not None and delay >= float(cfg["data_stale_minutes"]):
        add("data_stale", "warning", "Market data delay exceeds configured threshold", delay, cfg["data_stale_minutes"])

    scanner = metadata.get("rich_cheap_scanner") or {}
    candidates = scanner.get("candidates") or []
    if candidates:
        max_abs = max(abs(float(item.get("residual") or 0.0)) for item in candidates)
    else:
        max_abs = _finite_or_none(metadata.get("rich_cheap_max_abs_residual")) or 0.0
    if max_abs >= float(cfg["rich_cheap_residual_threshold"]):
        add("rich_cheap_residual", "info", "Rich/cheap residual exceeds configured threshold", max_abs, cfg["rich_cheap_residual_threshold"])

    if log_path is not None and alerts:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for alert in alerts:
                handle.write(json.dumps(alert, sort_keys=True) + "\n")

    return {
        "available": True,
        "source": "local_surface_alert_rules",
        "configured": cfg,
        "alert_count": len(alerts),
        "alerts": alerts,
        "log_path": str(log_path) if log_path is not None else None,
    }


def watchlist_presets(events: list[dict[str, Any]] | None = None, *, as_of: date | None = None) -> dict[str, list[str]]:
    """Return deterministic dashboard watchlist presets."""
    presets = {
        "Mega-cap tech": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA"],
        "Indices": ["SPY", "QQQ", "IWM", "VTI"],
        "High beta": ["TSLA", "AMD", "NVDA", "PLTR", "COIN", "SOFI", "HOOD", "DKNG"],
        "Financials": ["JPM", "BAC", "WFC", "GS", "V", "MA"],
        "Earnings this week": [],
    }
    today = as_of or date.today()
    horizon = (pd.Timestamp(today) + pd.Timedelta(days=7)).date()
    earnings = []
    for event in events or []:
        if str(event.get("event_type", "")).lower() != "earnings":
            continue
        event_date = pd.to_datetime(event.get("event_date"), errors="coerce")
        if pd.isna(event_date):
            continue
        if today <= event_date.date() <= horizon:
            earnings.append(str(event.get("symbol", "")).upper())
    presets["Earnings this week"] = sorted({symbol for symbol in earnings if symbol})
    return presets


def save_surface_workspace(
    workspace: dict[str, Any],
    directory: Path | str = "data/workspaces",
    *,
    name: str | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Persist selected symbols, filters, model settings, and chart layout as JSON."""
    if not isinstance(workspace, dict):
        return _unavailable("Workspace must be a dictionary")
    saved_at = timestamp or datetime.utcnow()
    workspace_name = name or str(workspace.get("name") or "workspace")
    payload = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "name": workspace_name,
        "saved_at": saved_at.isoformat(),
        "selected_symbols": [str(symbol).upper() for symbol in workspace.get("selected_symbols", [])],
        "filters": _json_safe(workspace.get("filters") or {}),
        "model_settings": _json_safe(workspace.get("model_settings") or {}),
        "chart_layout": _json_safe(workspace.get("chart_layout") or {}),
        "provenance": _json_safe(
            workspace.get("provenance")
            or {
                "source": "local_workspace_config",
                "created_by": "dashboard",
            }
        ),
    }
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_slugify(workspace_name)}_{saved_at.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "available": True,
        "source": "local_workspace_config",
        "path": str(path),
        "workspace": payload,
    }


def load_surface_workspace(path: Path | str) -> dict[str, Any]:
    """Load a workspace JSON file created by ``save_surface_workspace``."""
    workspace_path = Path(path)
    if not workspace_path.exists():
        return _unavailable(f"Workspace not found: {workspace_path}")
    payload = json.loads(workspace_path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "source": "local_workspace_config",
        "path": str(workspace_path),
        "workspace": payload,
    }


def list_surface_workspaces(directory: Path | str = "data/workspaces") -> list[dict[str, Any]]:
    """List saved workspace configs newest first without loading unrelated files."""
    root = Path(directory)
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "path": str(path),
                "name": payload.get("name") or path.stem,
                "saved_at": payload.get("saved_at"),
                "selected_symbols": payload.get("selected_symbols") or [],
                "provenance": payload.get("provenance") or {},
            }
        )
    return rows


def compare_saved_snapshots(left: Any, right: Any) -> dict[str, Any]:
    """Compare two saved snapshots across IV surface, skew, term, and scanner residuals."""
    left_payload = _snapshot_payload(left)
    right_payload = _snapshot_payload(right)
    left_frame = left_payload["frame"]
    right_frame = right_payload["frame"]
    if left_frame.empty or right_frame.empty:
        return _unavailable("Both snapshots need option rows")

    left_prepared = _comparison_frame(left_frame)
    right_prepared = _comparison_frame(right_frame)
    matched = left_prepared.merge(
        right_prepared,
        on=["expiration", "strike", "type"],
        suffixes=("_left", "_right"),
    )
    if matched.empty:
        return _unavailable("No matching expiration, strike, and type rows")

    matched["iv_delta"] = matched["iv_right"] - matched["iv_left"]
    matched["price_delta"] = matched["price_right"] - matched["price_left"]
    return {
        "available": True,
        "source": "saved_snapshot_comparison",
        "left": left_payload["provenance"],
        "right": right_payload["provenance"],
        "surface_deltas": {
            "matched_points": int(len(matched)),
            "mean_iv_delta": _finite_or_none(matched["iv_delta"].mean()),
            "median_abs_iv_delta": _finite_or_none(matched["iv_delta"].abs().median()),
            "max_abs_iv_delta": _finite_or_none(matched["iv_delta"].abs().max()),
            "mean_price_delta": _finite_or_none(matched["price_delta"].mean()),
        },
        "skew_deltas": _metric_deltas(_snapshot_skew(left_prepared), _snapshot_skew(right_prepared), "risk_reversal"),
        "term_deltas": _metric_deltas(_snapshot_term(left_prepared), _snapshot_term(right_prepared), "median_iv"),
        "scanner_deltas": _metric_deltas(_snapshot_scanner(left_prepared), _snapshot_scanner(right_prepared), "residual"),
        "matched_rows": matched.replace({np.nan: None}).to_dict("records"),
    }


def estimate_transaction_costs(
    trades: list[dict[str, Any]] | pd.DataFrame,
    *,
    commission_per_contract: float = 0.65,
    slippage_bps: float = 1.0,
    assignment_fee: float = 0.0,
    exercise_fee: float = 0.0,
    contract_multiplier: float = CONTRACT_MULTIPLIER,
) -> dict[str, Any]:
    """Estimate explicit spread, slippage, commission, assignment, and exercise costs."""
    details = []
    for trade in _coerce_records(trades):
        quantity = abs(_finite_or_none(trade.get("quantity") or trade.get("contracts")) or 0.0)
        price = _first_finite(trade, "price", "mark", "mid", "selectedMarketPrice") or 0.0
        bid = _finite_or_none(trade.get("bid"))
        ask = _finite_or_none(trade.get("ask"))
        spread = max((ask - bid) if bid is not None and ask is not None else 0.0, 0.0)
        multiplier = _finite_or_none(trade.get("contract_multiplier")) or float(contract_multiplier)
        action = str(trade.get("action") or trade.get("side") or "trade").lower()
        spread_cost = quantity * spread * 0.5 * multiplier
        slippage_cost = quantity * price * (float(slippage_bps) / 10000.0) * multiplier
        commission = quantity * float(commission_per_contract)
        assignment = quantity * float(assignment_fee) if action == "assignment" else 0.0
        exercise = quantity * float(exercise_fee) if action == "exercise" else 0.0
        total = spread_cost + slippage_cost + commission + assignment + exercise
        details.append(
            {
                "symbol": trade.get("symbol") or trade.get("contract") or "",
                "action": action,
                "quantity": float(quantity),
                "price": float(price),
                "spread_cost": float(spread_cost),
                "slippage_cost": float(slippage_cost),
                "commission": float(commission),
                "assignment_fee": float(assignment),
                "exercise_fee": float(exercise),
                "total_cost": float(total),
            }
        )
    return {
        "available": True,
        "source": "explicit_transaction_cost_model",
        "trade_count": len(details),
        "total_cost": float(sum(row["total_cost"] for row in details)),
        "spread_cost": float(sum(row["spread_cost"] for row in details)),
        "slippage_cost": float(sum(row["slippage_cost"] for row in details)),
        "commissions": float(sum(row["commission"] for row in details)),
        "assignment_exercise_fees": float(sum(row["assignment_fee"] + row["exercise_fee"] for row in details)),
        "details": details,
        "assumptions": {
            "commission_per_contract": float(commission_per_contract),
            "slippage_bps": float(slippage_bps),
            "assignment_fee": float(assignment_fee),
            "exercise_fee": float(exercise_fee),
            "contract_multiplier": float(contract_multiplier),
        },
    }


def run_signal_backtest(
    observations: list[dict[str, Any]] | pd.DataFrame,
    *,
    initial_cash: float = 100000.0,
    notional: float = 10000.0,
    cost_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Backtest deterministic IV-rank, skew, term-structure, and residual signals."""
    frame = pd.DataFrame(_coerce_records(observations))
    if frame.empty:
        return _unavailable("No observations supplied")
    if "date" not in frame.columns:
        return _unavailable("Backtest observations require a date column")
    price_col = "close" if "close" in frame.columns else "price"
    if price_col not in frame.columns:
        return _unavailable("Backtest observations require close or price")

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["price"] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.dropna(subset=["date", "price"]).sort_values("date").reset_index(drop=True)
    if len(work) < 2:
        return _unavailable("Backtest requires at least two dated prices")

    costs = cost_config or {}
    equity = float(initial_cash)
    previous_price = float(work.loc[0, "price"])
    previous_position = 0.0
    peak = equity
    rows = []
    returns = []
    winning_days = 0
    turnover = 0.0
    total_cost = 0.0
    for index, row in work.iterrows():
        price = float(row["price"])
        target_position, reason = _signal_position(row)
        gross_pnl = 0.0 if index == 0 else previous_position * float(notional) * ((price / previous_price) - 1.0)
        trade_notional = abs(target_position - previous_position) * float(notional)
        trade_cost = 0.0
        if trade_notional > 0.0:
            turnover += trade_notional
            cost_result = estimate_transaction_costs(
                [
                    {
                        "symbol": row.get("symbol", ""),
                        "action": "buy" if target_position > previous_position else "sell",
                        "quantity": trade_notional / max(price, 1e-9),
                        "price": price,
                        "bid": row.get("bid"),
                        "ask": row.get("ask"),
                    }
                ],
                contract_multiplier=1.0,
                **costs,
            )
            trade_cost = cost_result["total_cost"]
        net_pnl = gross_pnl - trade_cost
        equity += net_pnl
        daily_return = net_pnl / float(initial_cash)
        returns.append(daily_return)
        if net_pnl > 0.0:
            winning_days += 1
        peak = max(peak, equity)
        total_cost += trade_cost
        rows.append(
            {
                "date": row["date"].date().isoformat(),
                "price": price,
                "signal": reason,
                "target_position": float(target_position),
                "gross_pnl": float(gross_pnl),
                "transaction_costs": float(trade_cost),
                "net_pnl": float(net_pnl),
                "equity": float(equity),
                "drawdown": float((equity / peak) - 1.0 if peak else 0.0),
            }
        )
        previous_price = price
        previous_position = target_position

    returns_array = np.asarray(returns[1:] or returns, dtype=float)
    std = float(np.std(returns_array, ddof=0))
    sharpe = None if std <= 1e-12 else float(np.mean(returns_array) / std * np.sqrt(252.0))
    return {
        "available": True,
        "source": "deterministic_signal_backtest",
        "initial_cash": float(initial_cash),
        "final_equity": float(equity),
        "return": float((equity / float(initial_cash)) - 1.0),
        "max_drawdown": float(min(row["drawdown"] for row in rows)),
        "hit_rate": float(winning_days / len(rows)),
        "sharpe": sharpe,
        "turnover": float(turnover),
        "transaction_costs": float(total_cost),
        "rows": rows,
        "cost_model": estimate_transaction_costs([], contract_multiplier=1.0, **costs)["assumptions"],
    }


def paper_trading_simulator(
    orders: list[dict[str, Any]] | pd.DataFrame,
    marks: dict[str, Any] | pd.DataFrame,
    *,
    starting_cash: float = 100000.0,
    existing_state: dict[str, Any] | None = None,
    cost_config: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Enter, mark, and track simulated positions without broker connectivity."""
    state = existing_state or {}
    cash = float(state.get("cash", starting_cash))
    positions = {str(key).upper(): dict(value) for key, value in (state.get("positions") or {}).items()}
    realized_pnl = float(state.get("realized_pnl", 0.0))
    order_log = list(state.get("order_log") or [])
    mark_map = _mark_map(marks)
    for order in _coerce_records(orders):
        symbol = str(order.get("symbol") or "").upper()
        if not symbol:
            continue
        side = str(order.get("side") or order.get("action") or "buy").lower()
        signed_quantity = abs(_finite_or_none(order.get("quantity")) or 0.0) * (-1.0 if side == "sell" else 1.0)
        price = _first_finite(order, "price", "mark") or mark_map.get(symbol) or 0.0
        cost_result = estimate_transaction_costs(
            [{**order, "quantity": abs(signed_quantity), "price": price, "symbol": symbol}],
            **(cost_config or {}),
        )
        trade_cost = cost_result["total_cost"]
        position = positions.setdefault(symbol, {"quantity": 0.0, "avg_cost": 0.0})
        old_quantity = float(position.get("quantity") or 0.0)
        old_cost = float(position.get("avg_cost") or 0.0)
        new_quantity = old_quantity + signed_quantity
        if signed_quantity < 0.0 and old_quantity > 0.0:
            closed = min(abs(signed_quantity), old_quantity)
            realized_pnl += closed * (price - old_cost) * CONTRACT_MULTIPLIER - trade_cost
        if new_quantity == 0.0:
            positions.pop(symbol, None)
        else:
            if old_quantity == 0.0 or np.sign(old_quantity) == np.sign(signed_quantity):
                avg_cost = ((old_quantity * old_cost) + (signed_quantity * price)) / new_quantity
            else:
                avg_cost = old_cost if abs(new_quantity) < abs(old_quantity) else price
            position["quantity"] = float(new_quantity)
            position["avg_cost"] = float(avg_cost)
        cash -= signed_quantity * price * CONTRACT_MULTIPLIER + trade_cost
        order_log.append(
            {
                "timestamp": (timestamp or datetime.utcnow()).isoformat(),
                "symbol": symbol,
                "side": side,
                "quantity": float(abs(signed_quantity)),
                "price": float(price),
                "transaction_costs": float(trade_cost),
                "mode": "paper",
            }
        )

    marked_positions = []
    market_value = 0.0
    for symbol, position in sorted(positions.items()):
        mark = float(mark_map.get(symbol, position.get("avg_cost") or 0.0))
        quantity = float(position.get("quantity") or 0.0)
        value = quantity * mark * CONTRACT_MULTIPLIER
        market_value += value
        marked_positions.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "avg_cost": float(position.get("avg_cost") or 0.0),
                "mark": mark,
                "market_value": float(value),
                "unrealized_pnl": float(quantity * (mark - float(position.get("avg_cost") or 0.0)) * CONTRACT_MULTIPLIER),
            }
        )
    return {
        "available": True,
        "source": "local_paper_trading_simulator",
        "mode": "paper",
        "broker_required": False,
        "cash": float(cash),
        "market_value": float(market_value),
        "equity": float(cash + market_value),
        "realized_pnl": float(realized_pnl),
        "positions": marked_positions,
        "order_log": order_log,
        "trading_enabled": True,
    }


def broker_integration_abstraction(
    positions: list[dict[str, Any]] | pd.DataFrame | None = None,
    *,
    account: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose a read-only broker shape while keeping all live trading disabled."""
    return {
        "available": True,
        "source": "read_only_broker_abstraction",
        "account": _json_safe(account or {}),
        "positions": _json_safe(_coerce_records(positions if positions is not None else [])),
        "capabilities": {
            "read_positions": True,
            "read_balances": True,
            "place_orders": False,
            "cancel_orders": False,
            "exercise_options": False,
            "live_trading": False,
        },
        "trade_submission": {
            "enabled": False,
            "reason": "Live order actions are intentionally disabled until broker trading is explicitly designed.",
        },
    }


def export_analysis_notebook(
    analysis: dict[str, Any],
    path: Path | str,
    *,
    title: str = "Volatility Surface Analysis",
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Export the current analysis payload to a reproducible local Jupyter notebook."""
    exported_at = timestamp or datetime.utcnow()
    safe_analysis = _json_safe(analysis)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    f"- Exported at: {exported_at.isoformat()}\n",
                    f"- Data timestamp: {safe_analysis.get('data_timestamp', 'unknown')}\n",
                    f"- Model assumptions: {safe_analysis.get('model_assumptions', 'not supplied')}\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "analysis = json.loads('''",
                    json.dumps(safe_analysis, indent=2, sort_keys=True),
                    "''')\n",
                    "analysis\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "provenance": {"source": "local_notebook_export", "exported_at": exported_at.isoformat()},
        },
        "nbformat": NOTEBOOK_NBFORMAT,
        "nbformat_minor": 5,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "available": True,
        "source": "local_notebook_export",
        "path": str(output_path),
        "cell_count": len(notebook["cells"]),
        "data_timestamp": safe_analysis.get("data_timestamp"),
        "model_assumptions": safe_analysis.get("model_assumptions"),
    }


def _vol_profile(row: dict[str, Any]) -> dict[str, Any]:
    symbol = str(row.get("symbol") or row.get("Symbol") or "").upper()
    if not symbol:
        return {}
    atm_iv = _first_finite(row, "atm_iv", "iv_30d", "30D IV", "front_iv")
    realized = _first_finite(row, "realized_20d", "realized_20d_latest", "20D Realized")
    iv_realized = _first_finite(row, "iv_realized_spread", "IV-Realized")
    if iv_realized is None and atm_iv is not None and realized is not None:
        iv_realized = atm_iv - realized
    term_slope = _first_finite(row, "term_slope", "slope_per_30d")
    if term_slope is None:
        iv_90 = _first_finite(row, "iv_90d", "90D IV")
        if atm_iv is not None and iv_90 is not None:
            term_slope = iv_90 - atm_iv
    return {
        "symbol": symbol,
        "atm_iv": atm_iv,
        "iv_rank": _first_finite(row, "iv_rank", "IV Rank"),
        "iv_percentile": _first_finite(row, "iv_percentile", "IV Percentile"),
        "skew_25d": _first_finite(row, "skew_25d", "front_risk_reversal_25d", "25D Skew"),
        "term_slope": term_slope,
        "realized_20d": realized,
        "iv_realized_spread": iv_realized,
        "mode": row.get("mode") or row.get("Mode") or row.get("data_mode"),
        "source": row.get("source") or row.get("Source") or row.get("iv_source"),
    }


def _normalized_overlay_records(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    labels = {
        "atm_iv": "ATM IV",
        "skew_25d": "25D Risk Reversal",
        "term_slope": "Term Slope",
        "iv_realized_spread": "IV - Realized",
    }
    for key, label in labels.items():
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None or right_value is None:
            continue
        scale = max(abs(float(left_value)), abs(float(right_value)), 1e-9)
        records.append(
            {
                "metric": label,
                "left_symbol": left["symbol"],
                "right_symbol": right["symbol"],
                "left_value": float(left_value),
                "right_value": float(right_value),
                "spread": float(left_value) - float(right_value),
                "left_normalized": float(left_value) / scale,
                "right_normalized": float(right_value) / scale,
            }
        )
    return records


def _earnings_events(events: Any) -> list[dict[str, Any]]:
    if hasattr(events, "upcoming"):
        raw_events = [event.as_dict() for event in events.upcoming()]
    elif isinstance(events, dict):
        raw_events = events.get("events") or []
    else:
        raw_events = list(events or [])
    earnings = [event for event in raw_events if str(event.get("event_type", "")).lower() == "earnings"]
    return sorted(earnings, key=lambda item: str(item.get("event_date") or ""))


def _atm_iv_by_expiry(chain: pd.DataFrame, spot: float, iv_column: str) -> list[dict[str, Any]]:
    if chain.empty:
        return []
    column = iv_column if iv_column in chain else "impliedVolatility"
    required = {"expiration", "strike", "daysToExpiration", column}
    if not required.issubset(chain.columns):
        return []
    work = chain.copy()
    work["expiration_dt"] = pd.to_datetime(work["expiration"], errors="coerce")
    work["strike_num"] = pd.to_numeric(work["strike"], errors="coerce")
    work["dte_num"] = pd.to_numeric(work["daysToExpiration"], errors="coerce")
    work["iv_num"] = pd.to_numeric(work[column], errors="coerce")
    rows = []
    for expiry, group in work.dropna(subset=["expiration_dt", "strike_num", "iv_num"]).groupby("expiration_dt"):
        row = group.loc[(group["strike_num"] - spot).abs().idxmin()]
        rows.append(
            {
                "expiration": expiry.date().isoformat(),
                "dte": float(row["dte_num"]) if _is_finite(row["dte_num"]) else None,
                "atm_iv": float(row["iv_num"]),
            }
        )
    return sorted(rows, key=lambda item: item["expiration"])


def _post_event_crush(atm_term: list[dict[str, Any]], selected_expiry: pd.Timestamp) -> float | None:
    if len(atm_term) < 2:
        return None
    selected_key = selected_expiry.date().isoformat()
    for index, row in enumerate(atm_term):
        if row["expiration"] == selected_key and index + 1 < len(atm_term):
            next_iv = atm_term[index + 1].get("atm_iv")
            this_iv = row.get("atm_iv")
            if this_iv is not None and next_iv is not None:
                return max(float(this_iv) - float(next_iv), 0.0)
    return None


def _surface_axes(strike_grid: Any, expiry_grid: Any, surface: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if strike_grid is None or expiry_grid is None or surface is None:
        return np.array([]), np.array([]), np.array([])
    vols = np.asarray(surface, dtype=float)
    strikes_raw = np.asarray(strike_grid, dtype=float)
    expiries_raw = np.asarray(expiry_grid, dtype=float)
    if vols.ndim != 2:
        return np.array([]), np.array([]), np.array([])
    strikes = strikes_raw[0, :] if strikes_raw.ndim == 2 else strikes_raw
    expiries = expiries_raw[:, 0] if expiries_raw.ndim == 2 else expiries_raw
    order_x = np.argsort(strikes)
    order_y = np.argsort(expiries)
    return strikes[order_x], expiries[order_y], vols[np.ix_(order_y, order_x)]


def _prepared_chain(chain: pd.DataFrame) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    required = {"type", "expiration", "daysToExpiration", "strike"}
    if not required.issubset(chain.columns):
        return pd.DataFrame()
    out = chain.copy()
    out["type_norm"] = out["type"].astype(str).str.lower()
    out["expiration_dt"] = pd.to_datetime(out["expiration"], errors="coerce")
    out["strike_num"] = pd.to_numeric(out["strike"], errors="coerce")
    out["dte_num"] = pd.to_numeric(out["daysToExpiration"], errors="coerce")
    return out.dropna(subset=["expiration_dt", "strike_num", "dte_num"])


def _strategy_template(chain: pd.DataFrame, spot: float, strategy_type: str) -> list[dict[str, Any]]:
    key = _strategy_key(strategy_type)
    near = _nearest_expiry(chain, 30.0)
    far = _next_expiry(chain, near)
    if near is None:
        return []
    if key == "straddle":
        return [_leg("call", near, spot, 1), _leg("put", near, spot, 1)]
    if key == "strangle":
        return [_leg("put", near, spot * 0.95, 1), _leg("call", near, spot * 1.05, 1)]
    if key == "vertical":
        strikes = _expiry_strikes(chain, near, "call")
        lower, upper = _adjacent_strikes(strikes, spot)
        return [] if lower is None or upper is None else [_leg("call", near, lower, 1), _leg("call", near, upper, -1)]
    if key == "calendar" and far is not None:
        return [_leg("call", near, spot, -1), _leg("call", far, spot, 1)]
    if key == "diagonal" and far is not None:
        far_strike = _above_strike(_expiry_strikes(chain, far, "call"), spot)
        return [] if far_strike is None else [_leg("call", near, spot, -1), _leg("call", far, far_strike, 1)]
    if key == "butterfly":
        strikes = _expiry_strikes(chain, near, "call")
        triple = _centered_strikes(strikes, spot, 3)
        return [] if len(triple) < 3 else [
            _leg("call", near, triple[0], 1),
            _leg("call", near, triple[1], -2),
            _leg("call", near, triple[2], 1),
        ]
    if key == "condor":
        strikes = _centered_strikes(_expiry_strikes(chain, near, "call"), spot, 4)
        return [] if len(strikes) < 4 else [
            _leg("call", near, strikes[0], 1),
            _leg("call", near, strikes[1], -1),
            _leg("call", near, strikes[2], -1),
            _leg("call", near, strikes[3], 1),
        ]
    if key == "risk_reversal":
        put_strike = _below_strike(_expiry_strikes(chain, near, "put"), spot)
        call_strike = _above_strike(_expiry_strikes(chain, near, "call"), spot)
        return [] if put_strike is None or call_strike is None else [
            _leg("put", near, put_strike, -1),
            _leg("call", near, call_strike, 1),
        ]
    return []


def _price_leg(
    row: pd.Series,
    spot: float,
    leg: dict[str, Any],
    strike_grid: Any | None,
    expiry_grid: Any | None,
    surface: Any | None,
) -> dict[str, Any]:
    strike = float(row["strike_num"])
    dte = float(row["dte_num"])
    option_type = str(row["type_norm"])
    surface_iv = surface_iv_for_contract(strike_grid, expiry_grid, surface, strike, dte)
    fallback_iv = _first_finite(row, "computedIV", "impliedVolatility")
    iv = surface_iv if surface_iv is not None else fallback_iv
    if iv is None or iv <= 0.0 or spot <= 0.0 or strike <= 0.0:
        return {}
    rate = _first_finite(row, "riskFreeRate") or 0.0
    dividend = _first_finite(row, "effectiveDividendYield", "dividendYield") or 0.0
    t = max(dte / 365.0, 0.0)
    model_price = BlackScholesModel.option_price(spot, strike, t, rate, iv, option_type, dividend)
    market_price = _first_finite(row, "selectedMarketPrice", "mark", "mid", "last")
    quantity = float(leg.get("quantity", 1.0))
    return {
        "contract": row.get("contractSymbol"),
        "type": option_type,
        "expiration": row["expiration_dt"].date().isoformat(),
        "dte": dte,
        "strike": strike,
        "quantity": quantity,
        "surface_iv": surface_iv,
        "pricing_iv": float(iv),
        "model_price": float(model_price),
        "market_price": market_price,
        "risk_free_rate": rate,
        "dividend_yield": dividend,
        "delta": float(OptionGreeks.delta(spot, strike, t, rate, iv, option_type, dividend)),
        "gamma": float(OptionGreeks.gamma(spot, strike, t, rate, iv, dividend)),
        "theta": float(OptionGreeks.theta(spot, strike, t, rate, iv, option_type, dividend)),
        "vega": float(OptionGreeks.vega(spot, strike, t, rate, iv, dividend)),
    }


def _strategy_payoff(legs: list[dict[str, Any]], spot: float, net_debit: float) -> dict[str, Any]:
    strikes = [item["strike"] for item in legs]
    low = max(0.01, min(strikes + [spot]) * 0.65)
    high = max(strikes + [spot]) * 1.35
    grid = np.linspace(low, high, 81)
    rows = []
    for terminal in grid:
        payoff = 0.0
        for leg in legs:
            intrinsic = max(terminal - leg["strike"], 0.0) if leg["type"] == "call" else max(leg["strike"] - terminal, 0.0)
            payoff += leg["quantity"] * intrinsic
        pnl = payoff - net_debit
        rows.append({"spot": float(terminal), "payoff": float(payoff), "pnl": float(pnl)})
    pnl_values = np.array([row["pnl"] for row in rows], dtype=float)
    return {
        "payoff_points": rows,
        "breakevens": _breakevens(rows),
        "max_profit": float(np.max(pnl_values)),
        "max_loss": float(np.min(pnl_values)),
        "max_profit_100x": float(np.max(pnl_values) * CONTRACT_MULTIPLIER),
        "max_loss_100x": float(np.min(pnl_values) * CONTRACT_MULTIPLIER),
    }


def _strategy_value(
    legs: list[dict[str, Any]],
    shocked_spot: float,
    *,
    time_pass: float,
    vol_shift: float,
    skew_shift: float,
    base_spot: float | None = None,
) -> float:
    value = 0.0
    reference_spot = base_spot or shocked_spot
    for leg in legs:
        strike = float(leg["strike"])
        dte = max(float(leg.get("dte") or 0.0) - float(time_pass), 0.0)
        t = dte / 365.0
        option_type = str(leg.get("type", "")).lower()
        quantity = float(leg.get("quantity") or 0.0)
        if t <= 0.0:
            price = max(shocked_spot - strike, 0.0) if option_type == "call" else max(strike - shocked_spot, 0.0)
        else:
            base_iv = float(leg.get("pricing_iv") or leg.get("surface_iv") or 0.0)
            log_money = np.log(strike / reference_spot) if reference_spot > 0 else 0.0
            skew_adjustment = -float(skew_shift) * float(np.clip(log_money, -0.30, 0.30)) / 0.30
            iv = max(0.01, min(5.0, base_iv + float(vol_shift) + skew_adjustment))
            rate = float(leg.get("risk_free_rate") or leg.get("riskFreeRate") or 0.0)
            dividend = float(leg.get("dividend_yield") or leg.get("effectiveDividendYield") or 0.0)
            price = BlackScholesModel.option_price(shocked_spot, strike, t, rate, iv, option_type, dividend)
        value += quantity * price
    return float(value)


def _aggregate_position_greeks(positions: list[dict[str, Any]]) -> dict[str, float]:
    totals = {
        "market_value_100x": 0.0,
        "cost_basis_100x": 0.0,
        "unrealized_pnl_100x": 0.0,
        "delta_100x": 0.0,
        "gamma_100x": 0.0,
        "theta_100x": 0.0,
        "vega_100x": 0.0,
    }
    for position in positions:
        quantity = float(position.get("quantity") or 0.0)
        totals["market_value_100x"] += float(position.get("market_value_100x") or 0.0)
        totals["cost_basis_100x"] += float(position.get("cost_basis_100x") or 0.0)
        totals["unrealized_pnl_100x"] += float(position.get("unrealized_pnl_100x") or 0.0)
        for greek in ("delta", "gamma", "theta", "vega"):
            totals[f"{greek}_100x"] += quantity * float(position.get(greek) or 0.0) * CONTRACT_MULTIPLIER
    return {key: float(value) for key, value in totals.items()}


def _portfolio_scenarios(
    positions: list[dict[str, Any]],
    spot_shifts: list[float] | tuple[float, ...],
    vol_shifts: list[float] | tuple[float, ...],
) -> list[dict[str, Any]]:
    rows = []
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for position in positions:
        by_symbol.setdefault(str(position.get("symbol", "")).upper(), []).append(position)

    for spot_shift in spot_shifts:
        for vol_shift in vol_shifts:
            pnl = 0.0
            for symbol_positions in by_symbol.values():
                if not symbol_positions:
                    continue
                reference_spot = _infer_symbol_spot(symbol_positions)
                for position in symbol_positions:
                    shocked_spot = reference_spot * (1.0 + float(spot_shift))
                    current = float(position.get("model_price") or 0.0)
                    unit_position = {**position, "quantity": 1.0}
                    shocked = _strategy_value(
                        [unit_position],
                        shocked_spot,
                        time_pass=0.0,
                        vol_shift=float(vol_shift),
                        skew_shift=0.0,
                        base_spot=reference_spot,
                    )
                    quantity = float(position.get("quantity") or 0.0)
                    pnl += quantity * (shocked - current) * CONTRACT_MULTIPLIER
            rows.append(
                {
                    "spot_shift": float(spot_shift),
                    "vol_shift": float(vol_shift),
                    "pnl_100x": float(pnl),
                }
            )
    return rows


def _infer_symbol_spot(positions: list[dict[str, Any]]) -> float:
    spots = [float(item.get("spot") or 0.0) for item in positions if _is_finite(item.get("spot"))]
    if spots:
        return float(np.median(spots))
    strikes = [float(item.get("strike") or 0.0) for item in positions if _is_finite(item.get("strike"))]
    if not strikes:
        return 100.0
    return float(np.median(strikes))


def _hedge_tradeoffs(contract: dict[str, Any], metric: str) -> str:
    greek = metric.replace("_100x", "")
    offset = {
        "delta": "directional exposure",
        "vega": "volatility exposure",
        "theta": "daily carry",
    }.get(greek, "target exposure")
    return (
        f"Uses the listed {contract.get('type')} as a single-contract hedge for {offset}; "
        "residual risk remains in gamma, skew, liquidity, and expiry basis."
    )


def _breakevens(rows: list[dict[str, float]]) -> list[float]:
    breakevens = []
    for left, right in zip(rows, rows[1:]):
        left_pnl = left["pnl"]
        right_pnl = right["pnl"]
        if left_pnl == 0.0:
            breakevens.append(left["spot"])
        if left_pnl * right_pnl < 0.0:
            weight = abs(left_pnl) / (abs(left_pnl) + abs(right_pnl))
            breakevens.append(left["spot"] + (right["spot"] - left["spot"]) * weight)
    return [float(value) for value in breakevens]


def _match_leg(chain: pd.DataFrame, leg: dict[str, Any]) -> pd.Series | None:
    option_type = str(leg.get("type", "")).lower()
    expiry = pd.to_datetime(leg.get("expiration"), errors="coerce")
    strike = _finite_or_none(leg.get("strike"))
    if pd.isna(expiry) or strike is None:
        return None
    side = chain[chain["type_norm"] == option_type].copy()
    if side.empty:
        return None
    side["expiry_distance"] = (side["expiration_dt"] - expiry).abs().dt.days
    side["strike_distance"] = (side["strike_num"] - strike).abs()
    return side.sort_values(["expiry_distance", "strike_distance"]).iloc[0]


def _nearest_expiry(chain: pd.DataFrame, target_dte: float) -> date | None:
    if chain.empty:
        return None
    idx = (chain["dte_num"] - target_dte).abs().idxmin()
    return chain.loc[idx, "expiration_dt"].date()


def _next_expiry(chain: pd.DataFrame, expiry: date | None) -> date | None:
    if expiry is None:
        return None
    expiries = sorted(pd.Timestamp(item).date() for item in chain["expiration_dt"].dropna().unique())
    for candidate in expiries:
        if candidate > expiry:
            return candidate
    return None


def _expiry_strikes(chain: pd.DataFrame, expiry: date, option_type: str) -> list[float]:
    rows = chain[(chain["expiration_dt"].dt.date == expiry) & (chain["type_norm"] == option_type)]
    return sorted(float(value) for value in rows["strike_num"].dropna().unique())


def _adjacent_strikes(strikes: list[float], spot: float) -> tuple[float | None, float | None]:
    if len(strikes) < 2:
        return None, None
    lower = min(strikes, key=lambda value: abs(value - spot))
    uppers = [value for value in strikes if value > lower]
    return lower, (uppers[0] if uppers else None)


def _centered_strikes(strikes: list[float], spot: float, count: int) -> list[float]:
    if len(strikes) < count:
        return []
    ordered = sorted(strikes, key=lambda value: abs(value - spot))[:count]
    return sorted(ordered)


def _above_strike(strikes: list[float], spot: float) -> float | None:
    above = [value for value in strikes if value > spot]
    return above[0] if above else None


def _below_strike(strikes: list[float], spot: float) -> float | None:
    below = [value for value in strikes if value < spot]
    return below[-1] if below else None


def _leg(option_type: str, expiry: date, strike: float, quantity: float) -> dict[str, Any]:
    return {"type": option_type, "expiration": expiry, "strike": strike, "quantity": quantity}


def _strategy_key(strategy_type: str) -> str:
    return str(strategy_type or "custom").strip().lower().replace(" ", "_").replace("-", "_")


def _spread(left: Any, right: Any) -> float | None:
    left_value = _finite_or_none(left)
    right_value = _finite_or_none(right)
    return None if left_value is None or right_value is None else left_value - right_value


def _first_finite(row: Any, *names: str) -> float | None:
    for name in names:
        value = row.get(name) if hasattr(row, "get") else None
        parsed = _finite_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _is_finite(value: Any) -> bool:
    return _finite_or_none(value) is not None


def _zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_json_safe(row) for row in value.replace({np.nan: None}).to_dict("records")]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in str(value).strip())
    return "_".join(part for part in slug.split("_") if part) or "workspace"


def _coerce_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.replace({np.nan: None}).to_dict("records")
    if isinstance(value, dict):
        return [value]
    return [dict(item) for item in value]


def _snapshot_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        from src.data.snapshots import load_snapshot

        snapshot = load_snapshot(value)
        timestamp = getattr(snapshot, "spot_timestamp", None)
        return {
            "frame": snapshot.options_frame(),
            "provenance": {
                "path": str(value),
                "symbol": snapshot.symbol,
                "timestamp": timestamp.isoformat() if timestamp else None,
                "source": snapshot.source,
                "mode": snapshot.mode,
            },
        }
    if hasattr(value, "options_frame"):
        timestamp = getattr(value, "spot_timestamp", None)
        return {
            "frame": value.options_frame(),
            "provenance": {
                "symbol": getattr(value, "symbol", None),
                "timestamp": timestamp.isoformat() if timestamp else None,
                "source": getattr(value, "source", None),
                "mode": getattr(value, "mode", None),
            },
        }
    if isinstance(value, pd.DataFrame):
        return {"frame": value.copy(), "provenance": {"source": "dataframe"}}
    if isinstance(value, dict):
        frame_value = []
        for key in ("options", "chain", "frame"):
            if key in value and value[key] is not None:
                frame_value = value[key]
                break
        frame = frame_value.copy() if isinstance(frame_value, pd.DataFrame) else pd.DataFrame(frame_value)
        provenance = value.get("provenance") or {key: value.get(key) for key in ("symbol", "timestamp", "source", "mode")}
        return {"frame": frame, "provenance": _json_safe(provenance)}
    return {"frame": pd.DataFrame(), "provenance": {}}


def _comparison_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["expiration"] = pd.to_datetime(work.get("expiration"), errors="coerce").dt.date.astype(str)
    work["strike"] = pd.to_numeric(work.get("strike"), errors="coerce")
    work["type"] = work.get("type", "").astype(str).str.lower()
    work["iv"] = _numeric_first(work, ("computedIV", "impliedVolatility", "pricing_iv", "iv"))
    work["price"] = _numeric_first(work, ("selectedMarketPrice", "mark", "mid", "last", "price"))
    work["dte"] = _numeric_first(work, ("daysToExpiration", "dte"))
    work["residual"] = _numeric_first(work, ("residual", "richCheapResidual", "modelResidual", "fittedResidual"))
    return work.dropna(subset=["expiration", "strike", "type", "iv"])[
        ["expiration", "strike", "type", "iv", "price", "dte", "residual"]
    ]


def _numeric_first(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for name in names:
        if name in frame:
            out = out.fillna(pd.to_numeric(frame[name], errors="coerce"))
    return out


def _snapshot_skew(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for expiry, group in frame.groupby("expiration"):
        calls = group[group["type"] == "call"]
        puts = group[group["type"] == "put"]
        common = sorted(set(calls["strike"]).intersection(set(puts["strike"])))
        if not common:
            continue
        strike = min(common, key=lambda value: abs(value - group["strike"].median()))
        call_iv = calls.loc[calls["strike"] == strike, "iv"].mean()
        put_iv = puts.loc[puts["strike"] == strike, "iv"].mean()
        rows.append({"expiration": expiry, "risk_reversal": _finite_or_none(call_iv - put_iv)})
    return rows


def _snapshot_term(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for expiry, group in frame.groupby("expiration"):
        rows.append(
            {
                "expiration": expiry,
                "dte": _finite_or_none(group["dte"].median()),
                "median_iv": _finite_or_none(group["iv"].median()),
            }
        )
    return sorted(rows, key=lambda item: (item.get("dte") is None, item.get("dte") or 0.0, item["expiration"]))


def _snapshot_scanner(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    residuals = frame.dropna(subset=["residual"])
    for expiry, group in residuals.groupby("expiration"):
        idx = group["residual"].abs().idxmax()
        row = group.loc[idx]
        rows.append(
            {
                "expiration": expiry,
                "strike": float(row["strike"]),
                "type": row["type"],
                "residual": float(row["residual"]),
            }
        )
    return rows


def _metric_deltas(left: list[dict[str, Any]], right: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    left_by_key = {item["expiration"]: item for item in left}
    right_by_key = {item["expiration"]: item for item in right}
    rows = []
    for key in sorted(set(left_by_key).intersection(right_by_key)):
        left_value = _finite_or_none(left_by_key[key].get(metric))
        right_value = _finite_or_none(right_by_key[key].get(metric))
        if left_value is None or right_value is None:
            continue
        rows.append(
            {
                "expiration": key,
                f"left_{metric}": left_value,
                f"right_{metric}": right_value,
                f"{metric}_delta": right_value - left_value,
            }
        )
    return rows


def _signal_position(row: pd.Series) -> tuple[float, str]:
    iv_rank = _finite_or_none(row.get("iv_rank"))
    skew = _first_finite(row, "skew_25d", "front_risk_reversal_25d")
    term = _first_finite(row, "term_slope", "term_structure")
    residual = _first_finite(row, "residual", "rich_cheap_residual")
    if residual is not None and residual >= 0.08:
        return -1.0, "rich residual"
    if residual is not None and residual <= -0.08:
        return 1.0, "cheap residual"
    if iv_rank is not None and iv_rank >= 0.75:
        return -1.0, "high IV rank"
    if iv_rank is not None and iv_rank <= 0.25:
        return 1.0, "low IV rank"
    if skew is not None and skew <= -0.06:
        return 0.5, "steep downside skew"
    if term is not None and term >= 0.05:
        return -0.5, "steep contango"
    return 0.0, "flat"


def _mark_map(marks: Any) -> dict[str, float]:
    if isinstance(marks, pd.DataFrame):
        records = marks.to_dict("records")
    elif isinstance(marks, dict):
        records = [{"symbol": key, "mark": value} for key, value in marks.items()]
    else:
        records = list(marks or [])
    out = {}
    for row in records:
        symbol = str(row.get("symbol") or "").upper()
        mark = _first_finite(row, "mark", "price", "close")
        if symbol and mark is not None:
            out[symbol] = mark
    return out


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}
