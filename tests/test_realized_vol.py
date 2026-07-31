import numpy as np
import pandas as pd

from src.marketdata.realized_vol import latest_realized_volatility, realized_volatility_estimators


def test_realized_volatility_estimators_return_annualized_ohlc_metrics():
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    close = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.18, len(dates))), index=dates)
    frame = pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0] * 0.995),
            "High": close * 1.015,
            "Low": close * 0.985,
            "Close": close,
        }
    )

    estimates = realized_volatility_estimators(frame, windows=(20, 60))
    latest = latest_realized_volatility(estimates)

    assert set(estimates) == {20, 60}
    assert estimates[20]["close_to_close"].notna().any()
    assert estimates[20]["parkinson"].dropna().iloc[-1] > 0.0
    assert estimates[60]["yang_zhang"].dropna().iloc[-1] > 0.0
    assert latest["garman_klass_20d"] is not None
    assert latest["rogers_satchell_60d"] is not None
