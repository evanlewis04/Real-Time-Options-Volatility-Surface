"""Phase 4 relative-value, event, and strategy analytics."""

from __future__ import annotations

from datetime import date
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
