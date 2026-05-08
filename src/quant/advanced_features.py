"""Phase 4 relative-value, event, and strategy analytics."""

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


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}
