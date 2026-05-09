import json
import subprocess
import sys

import pandas as pd

from dashboard_connector import DashboardConnector
from tests.fixtures.noisy_option_chain import (
    FIXTURE_SPOT,
    checked_clean_chain,
    checked_noisy_chain,
    fixture_reason_buckets,
    normalized_clean_chain,
    normalized_noisy_chain,
)
from src.quant.svi import calibrate_svi_by_expiry


def test_noisy_fixture_has_stable_row_counts_and_reason_buckets():
    chain = normalized_noisy_chain()
    checked, no_arb = checked_noisy_chain()

    assert len(chain) == 58
    assert len(checked) == 58
    assert chain.attrs["rejection_reasons"] == {
        "expired_contract": 1,
        "extreme_moneyness": 2,
        "old_quote": 1,
        "stale_last_only": 1,
        "wide_bid_ask_spread": 1,
    }
    assert chain.attrs["data_quality_score"] == 90.6
    assert int((chain["quoteQuality"] == "last_only").sum()) == 2
    assert float(pd.to_numeric(chain["impliedVolatility"], errors="coerce").max()) == 2.85
    assert no_arb["no_arbitrage_violation_rows"] == 15
    assert no_arb["no_arbitrage_reason_buckets"] == {
        "bounds": 1,
        "calendar_monotonicity": 6,
        "call_monotonicity": 2,
        "convexity": 4,
        "put_monotonicity": 1,
    }
    assert fixture_reason_buckets(chain, no_arb) == {
        "expired_contract": 1,
        "extreme_moneyness": 2,
        "no_arbitrage_violation": 15,
        "old_quote": 1,
        "stale_last_only": 1,
        "wide_bid_ask_spread": 1,
    }


def test_clean_fixture_is_high_quality_and_has_low_standard_fit_error():
    chain = normalized_clean_chain()
    checked, no_arb = checked_clean_chain()
    surface_chain = DashboardConnector._surface_iv_chain(checked)
    meta = DashboardConnector._svi_metadata(surface_chain, spot=200.0, iv_column="computedIV")

    assert len(chain) == 56
    assert chain.attrs["rejection_reasons"] == {}
    assert chain.attrs["data_quality_score"] == 100.0
    assert no_arb["no_arbitrage_violation_rows"] == 0
    assert len(surface_chain) == 56
    assert meta["fit_diagnostics"]["fitted_expiries"] == 4
    assert meta["fit_diagnostics"]["points"] == 56
    assert meta["fit_diagnostics"]["rmse"] < 0.005
    assert [row["mode"] for row in meta["fit_mode_comparison"]] == [
        "Standard SVI",
        "Robust SVI",
        "Robust SSVI",
    ]
    assert meta["fit_diagnostics"]["residual_diagnostics"]["policy"] == "diagnostic_only_no_rows_removed"


def test_robust_loss_improves_noisy_fixture_without_moving_clean_fixture():
    clean_checked, _ = checked_clean_chain()
    noisy_checked, _ = checked_noisy_chain()
    clean_surface_chain = DashboardConnector._surface_iv_chain(clean_checked)
    noisy_surface_chain = DashboardConnector._surface_iv_chain(noisy_checked)

    clean_linear = calibrate_svi_by_expiry(
        clean_surface_chain,
        spot=FIXTURE_SPOT,
        iv_column="computedIV",
        loss="linear",
    )
    clean_robust = calibrate_svi_by_expiry(
        clean_surface_chain,
        spot=FIXTURE_SPOT,
        iv_column="computedIV",
        loss="soft_l1",
    )
    noisy_linear = calibrate_svi_by_expiry(
        noisy_surface_chain,
        spot=FIXTURE_SPOT,
        iv_column="computedIV",
        loss="linear",
    )
    noisy_robust = calibrate_svi_by_expiry(
        noisy_surface_chain,
        spot=FIXTURE_SPOT,
        iv_column="computedIV",
        loss="soft_l1",
    )

    assert _mean_rmse(clean_robust) <= _mean_rmse(clean_linear) + 5e-6
    assert _mean_rmse(noisy_robust) < _mean_rmse(noisy_linear)
    assert _abs_residual_quantile(noisy_robust, 0.95) < _abs_residual_quantile(noisy_linear, 0.95)
    noisy_diagnostics = DashboardConnector._svi_metadata(noisy_surface_chain, spot=FIXTURE_SPOT, iv_column="computedIV")
    assert noisy_diagnostics["fit_diagnostics"]["residual_diagnostics"]["clipped_count"] >= 1
    assert noisy_diagnostics["fit_mode_comparison"][0]["mode"] == "Standard SVI"
    assert noisy_diagnostics["fit_mode_comparison"][0]["weight_mode"] == "uniform"
    assert noisy_diagnostics["fit_mode_comparison"][1]["mode"] == "Robust SVI"


def test_compare_surface_fit_modes_script_emits_deterministic_json():
    result = subprocess.run(
        [sys.executable, "scripts/compare_surface_fit_modes.py", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert [item["fixture"] for item in payload["fixtures"]] == ["clean", "noisy"]
    clean, noisy = payload["fixtures"]
    assert clean["raw_rows"] == 56
    assert clean["quality_score"] == 100.0
    assert clean["fit_rmse"] < 0.005
    assert noisy["raw_rows"] == 64
    assert noisy["normalized_rows"] == 58
    assert noisy["no_arbitrage_excluded_count"] == 15
    assert noisy["reason_buckets"]["no_arbitrage_violation"] == 15
    assert pd.notna(noisy["residual_quantiles"]["p95"])


def _mean_rmse(fitted: pd.DataFrame) -> float:
    return float(pd.to_numeric(fitted["rmse"], errors="coerce").mean())


def _abs_residual_quantile(fitted: pd.DataFrame, quantile: float) -> float:
    residuals = [
        abs(float(row["residual"]))
        for smile in fitted.to_dict("records")
        for row in smile.get("residuals") or []
    ]
    return float(pd.Series(residuals).quantile(quantile))
