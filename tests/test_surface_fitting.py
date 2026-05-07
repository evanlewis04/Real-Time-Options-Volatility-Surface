import numpy as np
import pandas as pd

from dashboard_connector import DashboardConnector
from src.quant.smoothing import smooth_iv_surface
from src.quant.svi import calibrate_svi_by_expiry, svi_total_variance


def test_smoothing_reduces_roughness_and_enforces_calendar_total_variance():
    strikes = np.array([90.0, 100.0, 110.0])
    expiries = np.array([30.0, 60.0, 90.0])
    vols = np.array(
        [
            [0.30, 0.50, 0.28],
            [0.20, 0.18, 0.19],
            [0.22, 0.24, 0.23],
        ]
    )

    smoothed, meta = smooth_iv_surface(strikes, expiries, vols)
    total_variance = smoothed**2 * (expiries.reshape(-1, 1) / 365.0)

    assert meta["applied"]
    assert meta["roughness_after"] < meta["roughness_before"]
    assert meta["calendar_adjustments"] > 0
    assert np.all(np.diff(total_variance, axis=0) >= -1e-10)


def test_svi_calibration_recovers_low_error_smile():
    expiry = pd.Timestamp("2026-06-19")
    dte = 45.0
    k = np.linspace(-0.20, 0.20, 9)
    params = {"a": 0.002, "b": 0.04, "rho": -0.35, "m": 0.02, "sigma": 0.30}
    total_variance = svi_total_variance(k, **params)
    iv = np.sqrt(total_variance / (dte / 365.0))
    chain = pd.DataFrame(
        {
            "expiration": expiry,
            "daysToExpiration": dte,
            "strike": 100.0 * np.exp(k),
            "logMoneyness": k,
            "computedIV": iv,
        }
    )

    fitted = calibrate_svi_by_expiry(chain, spot=100.0)

    assert len(fitted) == 1
    assert fitted.iloc[0]["points"] == 9
    assert fitted.iloc[0]["rmse"] < 1e-3
    assert len(fitted.iloc[0]["residuals"]) == 9


def test_connector_svi_metadata_includes_fit_diagnostics():
    expiry = pd.Timestamp("2026-06-19")
    dte = 45.0
    k = np.linspace(-0.20, 0.20, 9)
    total_variance = svi_total_variance(k, 0.002, 0.04, -0.35, 0.02, 0.30)
    chain = pd.DataFrame(
        {
            "expiration": expiry,
            "daysToExpiration": dte,
            "strike": 100.0 * np.exp(k),
            "logMoneyness": k,
            "computedIV": np.sqrt(total_variance / (dte / 365.0)),
        }
    )

    meta = DashboardConnector._svi_metadata(chain, spot=100.0)

    assert meta["fit_diagnostics"]["model"] == "SVI"
    assert meta["fit_diagnostics"]["fitted_expiries"] == 1
    assert meta["fit_diagnostics"]["points"] == 9
    assert meta["svi_smiles"][0]["rmse"] < 1e-3
