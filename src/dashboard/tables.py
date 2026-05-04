"""Table filtering and export helpers for dashboard grids."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


def filter_market_snapshot(
    frame: pd.DataFrame,
    modes: Iterable[str] | None = None,
    min_iv_30d: float | None = None,
) -> pd.DataFrame:
    """Filter the market snapshot grid without mutating the source frame."""
    if frame.empty:
        return frame.copy()

    filtered = frame.copy()
    mode_values = [mode for mode in (modes or []) if mode]
    if mode_values and "Mode" in filtered:
        filtered = filtered[filtered["Mode"].isin(mode_values)]

    if min_iv_30d is not None and "30D IV" in filtered:
        iv = pd.to_numeric(filtered["30D IV"], errors="coerce")
        filtered = filtered[iv.fillna(-1.0) >= min_iv_30d]

    return filtered


def filter_option_chain(
    frame: pd.DataFrame,
    max_spread_pct: float,
    min_open_interest: int,
    option_types: Iterable[str] | None = None,
    expirations: Iterable[object] | None = None,
    moneyness_range: Tuple[float, float] | None = None,
    iv_range: Tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Apply quote-quality and user-facing chain filters."""
    if frame.empty:
        return frame.copy()

    filtered = frame.copy()
    if "bidAskSpreadPct" in filtered:
        spread = pd.to_numeric(filtered["bidAskSpreadPct"], errors="coerce")
        filtered = filtered[spread.isna() | (spread <= max_spread_pct)]

    if "openInterest" in filtered:
        oi = pd.to_numeric(filtered["openInterest"], errors="coerce").fillna(0)
        filtered = filtered[oi >= min_open_interest]

    types = [str(value) for value in (option_types or []) if value]
    if types and "type" in filtered:
        filtered = filtered[filtered["type"].isin(types)]

    expiration_values = list(expirations or [])
    if expiration_values and "expiration" in filtered:
        selected = pd.to_datetime(pd.Series(expiration_values), errors="coerce").dt.normalize()
        expirations_norm = pd.to_datetime(filtered["expiration"], errors="coerce").dt.normalize()
        filtered = filtered[expirations_norm.isin(set(selected.dropna()))]

    if moneyness_range and "moneyness" in filtered:
        low, high = moneyness_range
        moneyness = pd.to_numeric(filtered["moneyness"], errors="coerce")
        filtered = filtered[(moneyness >= low) & (moneyness <= high)]

    if iv_range and "impliedVolatility" in filtered:
        low, high = iv_range
        iv = pd.to_numeric(filtered["impliedVolatility"], errors="coerce")
        filtered = filtered[(iv >= low) & (iv <= high)]

    return filtered


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a DataFrame for Streamlit download buttons."""
    return frame.to_csv(index=False).encode("utf-8")
