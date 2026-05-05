"""Delta-based skew and term-structure analytics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.pricing.black_scholes import OptionGreeks


def delta_skew_by_expiry(chain: pd.DataFrame, spot: float, iv_column: str = "computedIV") -> pd.DataFrame:
    """Return 10/25-delta put/call IV, ATM IV, risk reversal, and butterfly by expiry."""
    if chain.empty or spot <= 0 or "expiration" not in chain.columns:
        return pd.DataFrame()

    work = chain.copy()
    if iv_column not in work:
        iv_column = "impliedVolatility"
    required = {"type", "strike", "daysToExpiration", iv_column}
    if not required.issubset(work.columns):
        return pd.DataFrame()

    work["expiration_norm"] = pd.to_datetime(work["expiration"], errors="coerce").dt.normalize()
    work["iv_num"] = pd.to_numeric(work[iv_column], errors="coerce")
    work["strike_num"] = pd.to_numeric(work["strike"], errors="coerce")
    work["dte_num"] = pd.to_numeric(work["daysToExpiration"], errors="coerce")
    work["rate_num"] = _numeric_column(work, "riskFreeRate").fillna(0.0)
    dividend = _numeric_column(work, "effectiveDividendYield")
    if dividend.isna().all() and "dividendYield" in work:
        dividend = _numeric_column(work, "dividendYield")
    work["dividend_num"] = dividend.fillna(0.0)
    if "delta" in work:
        work["delta_num"] = pd.to_numeric(work["delta"], errors="coerce")
    else:
        work["delta_num"] = np.nan

    missing_delta = work["delta_num"].isna()
    if missing_delta.any():
        work.loc[missing_delta, "delta_num"] = [
            _row_delta(row, spot) for _, row in work.loc[missing_delta].iterrows()
        ]

    rows: list[dict[str, Any]] = []
    for expiry, group in work.dropna(subset=["expiration_norm"]).groupby("expiration_norm"):
        group = group.dropna(subset=["iv_num", "delta_num", "strike_num", "dte_num"])
        if group.empty:
            continue
        atm_row = group.loc[(group["strike_num"] - spot).abs().idxmin()]
        metrics: dict[str, Any] = {
            "expiration": expiry.date().isoformat(),
            "dte": float(group["dte_num"].median()),
            "atm_iv": float(atm_row["iv_num"]),
            "atm_strike": float(atm_row["strike_num"]),
        }
        for target in (0.10, 0.25):
            metrics[f"{int(target * 100)}d_put_iv"] = _nearest_delta_iv(group, "put", -target)
            metrics[f"{int(target * 100)}d_call_iv"] = _nearest_delta_iv(group, "call", target)

        put_25 = metrics.get("25d_put_iv")
        call_25 = metrics.get("25d_call_iv")
        atm = metrics.get("atm_iv")
        metrics["risk_reversal_25d"] = _spread(call_25, put_25)
        metrics["butterfly_25d"] = _butterfly(call_25, put_25, atm)
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values("dte").reset_index(drop=True) if rows else pd.DataFrame()


def term_structure_metrics(atm_term: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> dict[str, float | str | None]:
    """Summarize ATM term-structure slope, curvature, and regime."""
    clean = [(float(dte), float(iv)) for dte, iv in atm_term if np.isfinite(dte) and np.isfinite(iv)]
    if len(clean) < 2:
        return {
            "front_iv": clean[0][1] if clean else None,
            "back_iv": clean[0][1] if clean else None,
            "front_back_spread": None,
            "slope_per_30d": None,
            "curvature": None,
            "regime": "flat" if clean else "unavailable",
        }

    clean = sorted(clean)
    dte = np.array([item[0] for item in clean], dtype=float)
    iv = np.array([item[1] for item in clean], dtype=float)
    spread = float(iv[-1] - iv[0])
    slope = float(spread / max(dte[-1] - dte[0], 1.0) * 30.0)
    curvature = None
    if len(clean) >= 3:
        x = (dte - dte.mean()) / max(dte.std(), 1.0)
        curvature = float(np.polyfit(x, iv, 2)[0])
    if abs(spread) < 0.005:
        regime = "flat"
    else:
        regime = "contango" if spread > 0 else "backwardation"
    return {
        "front_iv": float(iv[0]),
        "back_iv": float(iv[-1]),
        "front_back_spread": spread,
        "slope_per_30d": slope,
        "curvature": curvature,
        "regime": regime,
    }


def _row_delta(row: pd.Series, spot: float) -> float:
    try:
        t = max(float(row["dte_num"]) / 365.0, 1e-9)
        sigma = float(row["iv_num"])
        if not np.isfinite(sigma) or sigma <= 0:
            return np.nan
        return float(
            OptionGreeks.delta(
                spot,
                float(row["strike_num"]),
                t,
                float(row["rate_num"]),
                sigma,
                str(row["type"]).lower(),
                float(row["dividend_num"]),
            )
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return np.nan


def _nearest_delta_iv(group: pd.DataFrame, option_type: str, target_delta: float) -> float | None:
    side = group[group["type"].astype(str).str.lower() == option_type].copy()
    side = side.dropna(subset=["delta_num", "iv_num"])
    if side.empty:
        return None
    idx = (side["delta_num"] - target_delta).abs().idxmin()
    return float(side.loc[idx, "iv_num"])


def _spread(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _butterfly(call_iv: Any, put_iv: Any, atm_iv: Any) -> float | None:
    if call_iv is None or put_iv is None or atm_iv is None:
        return None
    return float((call_iv + put_iv) / 2.0 - atm_iv)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")
