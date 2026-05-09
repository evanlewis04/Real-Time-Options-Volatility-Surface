import numpy as np
import pandas as pd

from dashboard_connector import DashboardConnector
from src.quant.smoothing import smooth_iv_surface
from src.quant.svi import calibrate_ssvi_surface, calibrate_svi_by_expiry, ssvi_total_variance, svi_total_variance


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
    assert fitted.iloc[0]["loss_mode"] == "soft_l1"
    assert fitted.iloc[0]["rmse"] < 1e-3
    assert len(fitted.iloc[0]["residuals"]) == 9


def test_svi_robust_loss_improves_clean_rows_around_noisy_center_point():
    expiry = pd.Timestamp("2026-06-19")
    dte = 45.0
    k = np.linspace(-0.35, 0.35, 15)
    params = {"a": 0.002, "b": 0.04, "rho": -0.35, "m": 0.02, "sigma": 0.30}
    clean_iv = np.sqrt(svi_total_variance(k, **params) / (dte / 365.0))
    observed_iv = clean_iv.copy()
    center_idx = len(k) // 2
    observed_iv[center_idx] *= 1.8
    chain = pd.DataFrame(
        {
            "expiration": expiry,
            "daysToExpiration": dte,
            "strike": 100.0 * np.exp(k),
            "logMoneyness": k,
            "computedIV": observed_iv,
        }
    )

    standard = calibrate_svi_by_expiry(chain, spot=100.0, weight_column=None, loss="linear")
    robust = calibrate_svi_by_expiry(chain, spot=100.0, weight_column=None, loss="huber", loss_f_scale=0.002)

    assert standard.iloc[0]["loss_mode"] == "linear"
    assert robust.iloc[0]["loss_mode"] == "huber"
    assert _clean_residual_rmse(standard.iloc[0]["residuals"], exclude_index=center_idx) > 0.02
    assert _clean_residual_rmse(robust.iloc[0]["residuals"], exclude_index=center_idx) < 0.005
    assert _abs_residual_quantile(robust.iloc[0]["residuals"], 0.90) < (
        _abs_residual_quantile(standard.iloc[0]["residuals"], 0.90) * 0.15
    )


def test_weighted_svi_calibration_prefers_high_weight_clean_rows():
    expiry = pd.Timestamp("2026-06-19")
    dte = 45.0
    k = np.linspace(-0.25, 0.25, 11)
    params = {"a": 0.002, "b": 0.04, "rho": -0.35, "m": 0.02, "sigma": 0.30}
    clean_iv = np.sqrt(svi_total_variance(k, **params) / (dte / 365.0))
    observed_iv = clean_iv.copy()
    observed_iv[-1] *= 1.45
    weights = np.ones(len(k))
    weights[-1] = 0.001
    chain = pd.DataFrame(
        {
            "expiration": expiry,
            "daysToExpiration": dte,
            "strike": 100.0 * np.exp(k),
            "logMoneyness": k,
            "computedIV": observed_iv,
            "fitWeight": weights,
        }
    )

    weighted = calibrate_svi_by_expiry(chain, spot=100.0)
    unweighted = calibrate_svi_by_expiry(chain, spot=100.0, weight_column=None)

    weighted_clean_rmse = _clean_residual_rmse(weighted.iloc[0]["residuals"])
    unweighted_clean_rmse = _clean_residual_rmse(unweighted.iloc[0]["residuals"])
    assert weighted.iloc[0]["weight_mode"] == "quote_reliability_liquidity"
    assert weighted.iloc[0]["positive_weight_count"] == 11
    assert weighted.iloc[0]["weighted_rmse"] < weighted.iloc[0]["rmse"]
    assert weighted_clean_rmse < unweighted_clean_rmse * 0.35


def test_ssvi_global_calibration_recovers_constrained_surface():
    expiries = [
        (pd.Timestamp("2026-06-19"), 45.0, 0.010),
        (pd.Timestamp("2026-08-21"), 108.0, 0.025),
        (pd.Timestamp("2026-11-20"), 199.0, 0.050),
    ]
    k = np.linspace(-0.25, 0.25, 11)
    rows = []
    for expiry, dte, theta in expiries:
        total_variance = ssvi_total_variance(k, theta, rho=-0.40, eta=1.10, gamma=0.25)
        for log_money, iv in zip(k, np.sqrt(total_variance / (dte / 365.0))):
            rows.append(
                {
                    "expiration": expiry,
                    "daysToExpiration": dte,
                    "strike": 100.0 * np.exp(log_money),
                    "logMoneyness": log_money,
                    "computedIV": iv,
                }
            )
    chain = pd.DataFrame(rows)

    fitted = calibrate_ssvi_surface(chain, spot=100.0)

    assert fitted["model"] == "SSVI"
    assert fitted["status"] == "fitted"
    assert fitted["loss_mode"] == "soft_l1"
    assert fitted["fitted_expiries"] == 3
    assert fitted["points"] == 33
    assert fitted["constraints"]["passed"]
    assert fitted["rmse"] < 1e-3
    assert len(fitted["residuals"]) == 33


def test_weighted_ssvi_calibration_reports_weighted_diagnostics():
    expiries = [
        (pd.Timestamp("2026-06-19"), 45.0, 0.010),
        (pd.Timestamp("2026-08-21"), 108.0, 0.025),
        (pd.Timestamp("2026-11-20"), 199.0, 0.050),
    ]
    k = np.linspace(-0.25, 0.25, 11)
    rows = []
    for expiry, dte, theta in expiries:
        total_variance = ssvi_total_variance(k, theta, rho=-0.40, eta=1.10, gamma=0.25)
        for log_money, clean_iv in zip(k, np.sqrt(total_variance / (dte / 365.0))):
            observed_iv = clean_iv
            weight = 1.0
            if dte == 108.0 and abs(log_money) < 1e-12:
                observed_iv = clean_iv * 1.6
                weight = 0.001
            rows.append(
                {
                    "expiration": expiry,
                    "daysToExpiration": dte,
                    "strike": 100.0 * np.exp(log_money),
                    "logMoneyness": log_money,
                    "computedIV": observed_iv,
                    "fitWeight": weight,
                }
            )
    chain = pd.DataFrame(rows)

    fitted = calibrate_ssvi_surface(chain, spot=100.0)
    diagnostics = {
        key: fitted[key]
        for key in ("weight_mode", "weight_column", "loss_mode", "weighted_rmse", "unweighted_rmse")
    }

    assert diagnostics["weight_mode"] == "quote_reliability_liquidity"
    assert diagnostics["weight_column"] == "fitWeight"
    assert diagnostics["loss_mode"] == "soft_l1"
    assert diagnostics["weighted_rmse"] < diagnostics["unweighted_rmse"] * 0.20
    assert fitted["constraints"]["passed"]
    assert min(row["fit_weight"] for row in fitted["residuals"]) == 0.001


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
    assert meta["fit_diagnostics"]["loss_mode"] == "soft_l1"
    assert meta["svi_smiles"][0]["rmse"] < 1e-3


def test_connector_svi_metadata_includes_global_ssvi_diagnostics():
    rows = []
    k = np.linspace(-0.20, 0.20, 9)
    for expiry, dte, theta in (
        (pd.Timestamp("2026-06-19"), 45.0, 0.012),
        (pd.Timestamp("2026-08-21"), 108.0, 0.028),
    ):
        total_variance = ssvi_total_variance(k, theta, rho=-0.30, eta=0.90, gamma=0.20)
        rows.extend(
            {
                "expiration": expiry,
                "daysToExpiration": dte,
                "strike": 100.0 * np.exp(log_money),
                "logMoneyness": log_money,
                "computedIV": iv,
            }
            for log_money, iv in zip(k, np.sqrt(total_variance / (dte / 365.0)))
        )
    chain = pd.DataFrame(rows)

    meta = DashboardConnector._svi_metadata(chain, spot=100.0)

    assert meta["global_fit_diagnostics"]["model"] == "SSVI"
    assert meta["global_fit_diagnostics"]["status"] == "fitted"
    assert meta["global_fit_diagnostics"]["fitted_expiries"] == 2
    assert meta["global_fit_diagnostics"]["loss_mode"] == "soft_l1"
    assert meta["global_fit_diagnostics"]["weight_mode"] == "uniform"
    assert meta["global_fit_diagnostics"]["weighted_rmse"] == meta["global_fit_diagnostics"]["unweighted_rmse"]
    assert meta["global_fit_diagnostics"]["constraints_passed"]
    assert meta["ssvi_surface"]["rmse"] < 1e-3


def _clean_residual_rmse(residuals, exclude_index: int | None = None):
    if exclude_index is None:
        values = [row["residual"] for row in residuals[:-1]]
    else:
        values = [row["residual"] for index, row in enumerate(residuals) if index != exclude_index]
    return float(np.sqrt(np.mean(np.square(values))))


def _abs_residual_quantile(residuals, quantile):
    values = [abs(row["residual"]) for row in residuals]
    return float(pd.Series(values).quantile(quantile))
