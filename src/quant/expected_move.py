"""Expected-move analytics from option chains."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def expected_moves_by_expiry(
    chain: pd.DataFrame,
    spot: float,
    *,
    iv_column: str = "computedIV",
    price_column: str = "selectedMarketPrice",
) -> pd.DataFrame:
    """Return expiry-level expected moves using ATM straddles or ATM IV."""
    if chain.empty or spot <= 0 or "expiration" not in chain.columns:
        return pd.DataFrame()

    work = chain.copy()
    if iv_column not in work:
        iv_column = "impliedVolatility"
    if price_column not in work:
        price_column = _fallback_price_column(work)

    required = {"type", "strike", "daysToExpiration", iv_column}
    if not required.issubset(work.columns):
        return pd.DataFrame()

    work["expiration_norm"] = pd.to_datetime(work["expiration"], errors="coerce").dt.normalize()
    work["strike_num"] = pd.to_numeric(work["strike"], errors="coerce")
    work["dte_num"] = pd.to_numeric(work["daysToExpiration"], errors="coerce")
    work["iv_num"] = pd.to_numeric(work[iv_column], errors="coerce")
    work["price_num"] = pd.to_numeric(work[price_column], errors="coerce") if price_column else np.nan
    work["type_norm"] = work["type"].astype(str).str.lower()
    work = work.dropna(subset=["expiration_norm", "strike_num", "dte_num"])

    rows: list[dict[str, Any]] = []
    for expiry, group in work.groupby("expiration_norm", dropna=True):
        dte_values = group["dte_num"].dropna()
        dte = float(dte_values.median()) if not dte_values.empty else np.nan
        if not np.isfinite(dte) or dte <= 0:
            continue

        atm_iv, iv_strike = _atm_iv(group, spot)
        straddle = _atm_straddle(group, spot)
        iv_move = _iv_expected_move(spot, atm_iv, dte)
        expected_move = straddle["move"] if straddle["move"] is not None else iv_move
        method = "atm_straddle" if straddle["move"] is not None else "atm_iv"
        if expected_move is None:
            continue

        expected_move = float(expected_move)
        rows.append(
            {
                "expiration": expiry.date().isoformat(),
                "dte": dte,
                "atm_strike": straddle["strike"] if straddle["strike"] is not None else iv_strike,
                "atm_iv": atm_iv,
                "call_price": straddle["call_price"],
                "put_price": straddle["put_price"],
                "straddle_move": straddle["move"],
                "iv_move": iv_move,
                "expected_move": expected_move,
                "expected_move_pct": expected_move / float(spot),
                "lower_bound": float(spot) - expected_move,
                "upper_bound": float(spot) + expected_move,
                "method": method,
                "source": price_column if method == "atm_straddle" else iv_column,
                "confidence": _confidence(method, straddle["move"], iv_move),
            }
        )

    return pd.DataFrame(rows).sort_values("dte").reset_index(drop=True) if rows else pd.DataFrame()


def _atm_straddle(group: pd.DataFrame, spot: float) -> dict[str, float | None]:
    pairs = []
    for strike, strike_group in group.groupby("strike_num", dropna=True):
        calls = strike_group[strike_group["type_norm"] == "call"]["price_num"].dropna()
        puts = strike_group[strike_group["type_norm"] == "put"]["price_num"].dropna()
        calls = calls[calls > 0]
        puts = puts[puts > 0]
        if calls.empty or puts.empty:
            continue
        call_price = float(calls.median())
        put_price = float(puts.median())
        pairs.append(
            {
                "strike": float(strike),
                "call_price": call_price,
                "put_price": put_price,
                "move": call_price + put_price,
                "atm_distance": abs(float(strike) - float(spot)),
            }
        )

    if not pairs:
        return {"strike": None, "call_price": None, "put_price": None, "move": None}
    best = sorted(pairs, key=lambda item: (item["atm_distance"], item["strike"]))[0]
    return {
        "strike": best["strike"],
        "call_price": best["call_price"],
        "put_price": best["put_price"],
        "move": best["move"],
    }


def _atm_iv(group: pd.DataFrame, spot: float) -> tuple[float | None, float | None]:
    ivs = group.dropna(subset=["iv_num", "strike_num"]).copy()
    ivs = ivs[(ivs["iv_num"] > 0) & np.isfinite(ivs["iv_num"])]
    if ivs.empty:
        return None, None
    ivs["atm_distance"] = (ivs["strike_num"] - float(spot)).abs()
    closest_distance = float(ivs["atm_distance"].min())
    sample = ivs[ivs["atm_distance"] == closest_distance]
    return float(sample["iv_num"].median()), float(sample.iloc[0]["strike_num"])


def _iv_expected_move(spot: float, atm_iv: float | None, dte: float) -> float | None:
    if atm_iv is None or not np.isfinite(atm_iv) or atm_iv <= 0 or dte <= 0:
        return None
    return float(spot) * float(atm_iv) * float(np.sqrt(dte / 365.0))


def _confidence(method: str, straddle_move: float | None, iv_move: float | None) -> str:
    if method == "atm_straddle" and straddle_move is not None and iv_move is not None:
        return "high"
    if method == "atm_straddle":
        return "medium"
    return "low"


def _fallback_price_column(frame: pd.DataFrame) -> str:
    for column in ("mark", "mid", "last"):
        if column in frame.columns:
            return column
    return ""
