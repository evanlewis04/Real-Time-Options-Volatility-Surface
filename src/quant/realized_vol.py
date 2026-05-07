"""Realized-volatility estimator suite."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


ESTIMATOR_COLUMNS = {
    "close_to_close": "Close-to-close",
    "parkinson": "Parkinson",
    "garman_klass": "Garman-Klass",
    "rogers_satchell": "Rogers-Satchell",
    "yang_zhang": "Yang-Zhang",
}


def realized_volatility_estimators(frame: pd.DataFrame, windows: Iterable[int] = (20, 60)) -> dict[int, pd.DataFrame]:
    """Calculate annualized realized-volatility estimators for OHLC data."""
    if frame.empty:
        return {int(window): _empty_frame() for window in windows}

    open_ = _positive_series(frame, "Open")
    high = _positive_series(frame, "High")
    low = _positive_series(frame, "Low")
    close = _positive_series(frame, "Close")
    log_close = np.log(close)
    close_return = log_close.diff()
    high_low = np.log(high / low)
    close_open = np.log(close / open_)
    high_close = np.log(high / close)
    high_open = np.log(high / open_)
    low_close = np.log(low / close)
    low_open = np.log(low / open_)
    overnight = np.log(open_ / close.shift(1))
    rs = high_close * high_open + low_close * low_open

    out: dict[int, pd.DataFrame] = {}
    for window in windows:
        size = int(window)
        if size <= 1:
            out[size] = _empty_frame()
            continue
        yz_weight = 0.34 / (1.34 + (size + 1.0) / (size - 1.0))
        estimates = pd.DataFrame(index=frame.index)
        estimates["close_to_close"] = close_return.rolling(size).std() * np.sqrt(252.0)
        estimates["parkinson"] = np.sqrt(_positive_variance((high_low**2).rolling(size).mean() * 252.0 / (4.0 * np.log(2.0))))
        estimates["garman_klass"] = np.sqrt(
            _positive_variance(
                (
                    0.5 * high_low**2
                    - (2.0 * np.log(2.0) - 1.0) * close_open**2
                ).rolling(size).mean()
                * 252.0
            )
        )
        estimates["rogers_satchell"] = np.sqrt(_positive_variance(rs.rolling(size).mean() * 252.0))
        yz_var = (
            overnight.rolling(size).var()
            + yz_weight * close_open.rolling(size).var()
            + (1.0 - yz_weight) * rs.rolling(size).mean()
        )
        estimates["yang_zhang"] = np.sqrt(_positive_variance(yz_var * 252.0))
        out[size] = estimates.replace([np.inf, -np.inf], np.nan)
    return out


def latest_realized_volatility(estimates: dict[int, pd.DataFrame]) -> dict[str, float | None]:
    """Return the latest available value for each estimator/window pair."""
    latest: dict[str, float | None] = {}
    for window, frame in sorted(estimates.items()):
        for column in ESTIMATOR_COLUMNS:
            series = frame[column].dropna() if column in frame else pd.Series(dtype=float)
            latest[f"{column}_{window}d"] = float(series.iloc[-1]) if not series.empty else None
    return latest


def _positive_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    series = pd.to_numeric(frame[column], errors="coerce")
    return series.where(series > 0)


def _positive_variance(series: pd.Series) -> pd.Series:
    return series.where(series >= 0.0)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ESTIMATOR_COLUMNS))
