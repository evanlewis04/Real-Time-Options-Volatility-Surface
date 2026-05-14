import json
import subprocess
import sys

import pandas as pd
import pytest

from tests.fixtures.noisy_option_chain import (
    FIXTURE_NOW,
    FIXTURE_SPOT,
    checked_clean_chain,
    checked_noisy_chain,
    fixture_reason_buckets,
)
from src.quant.surface_validation import backtest_fit_modes, fixture_snapshot_record, validate_fit_modes


def test_fit_mode_validation_metrics_run_offline_on_fixtures():
    clean, _ = checked_clean_chain()
    noisy, _ = checked_noisy_chain()

    validation = validate_fit_modes(noisy, FIXTURE_SPOT, baseline_chain=clean)

    assert validation["available"] is True
    assert validation["provenance"] == "fit_mode_validation_diagnostic_not_market_observation"
    assert validation["train_rows"] > validation["holdout_rows"] > 0
    modes = {row["mode"]: row for row in validation["modes"]}
    assert {"Standard SVI", "Robust SVI", "Robust SSVI"} <= set(modes)
    robust = modes["Robust SVI"]
    assert robust["oos_rmse"] > 0
    assert robust["residual_quantiles"]["p95"] > 0
    assert robust["out_of_sample_residuals_by_expiry"]
    assert robust["stability_vs_prior_day"]["available"] is True
    assert robust["no_arbitrage_violation_rate"] is not None
    assert robust["smoothness_penalty"]["adjacent_pair_count"] > 0


def test_backtest_fit_modes_flags_quality_driven_stability_improvement():
    clean, clean_no_arb = checked_clean_chain()
    noisy, noisy_no_arb = checked_noisy_chain()
    records = [
        fixture_snapshot_record(
            "clean",
            clean,
            FIXTURE_SPOT,
            FIXTURE_NOW,
            quality_score=float(clean.attrs["data_quality_score"]),
            reason_buckets=fixture_reason_buckets(clean, clean_no_arb),
        ),
        fixture_snapshot_record(
            "noisy",
            noisy,
            FIXTURE_SPOT,
            FIXTURE_NOW + pd.Timedelta(days=1),
            quality_score=float(noisy.attrs["data_quality_score"]),
            reason_buckets=fixture_reason_buckets(noisy, noisy_no_arb),
        ),
    ]

    report = backtest_fit_modes(records)

    assert report["available"] is True
    assert report["provenance"] == "fit_mode_backtest_diagnostic_not_market_observation"
    transition = report["transitions"][0]
    assert transition["quality_change"]["deteriorated_buckets"]["no_arbitrage_violation"] == 15
    assert transition["prior_assisted"]["provenance"] == "prior_assisted_fit_estimate_not_market_observation"
    assert "market observations" in report["estimate_warning"]


def test_backtest_reports_possible_hidden_real_move_when_quality_is_stable():
    clean, clean_no_arb = checked_clean_chain()
    shifted = clean.copy()
    shifted_mask = (
        (pd.to_datetime(shifted["expiration"]).dt.strftime("%Y-%m-%d") == "2026-06-07")
        & pd.to_numeric(shifted["strike"], errors="coerce").between(185.0, 215.0)
    )
    shifted.loc[shifted_mask, "computedIV"] = pd.to_numeric(shifted.loc[shifted_mask, "computedIV"], errors="coerce") + 0.25
    shifted.loc[shifted_mask, "impliedVolatility"] = shifted.loc[shifted_mask, "computedIV"]
    shifted["fitWeight"] = 1.0
    shifted.loc[shifted_mask, "fitWeight"] = 0.001
    buckets = fixture_reason_buckets(clean, clean_no_arb)
    records = [
        fixture_snapshot_record("clean", clean, FIXTURE_SPOT, FIXTURE_NOW, quality_score=100.0, reason_buckets=buckets),
        fixture_snapshot_record(
            "shifted",
            shifted,
            FIXTURE_SPOT,
            FIXTURE_NOW + pd.Timedelta(days=1),
            quality_score=100.0,
            reason_buckets=buckets,
        ),
    ]

    report = backtest_fit_modes(records)

    assert report["available"] is True
    assert report["transitions"][0]["quality_change"]["quality_deteriorated"] is False
    assert report["hides_real_move_risk_count"] == 1
    assert "Review" in report["transitions"][0]["interpretation"]


def test_validation_script_emits_deterministic_json():
    result = subprocess.run(
        [sys.executable, "scripts/validate_surface_fit_modes.py", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["validation"]["clean"]["available"] is True
    assert payload["validation"]["noisy"]["available"] is True
    assert payload["backtest"]["transition_count"] == 1
    assert payload["backtest"]["transitions"][0]["to"] == "noisy"
    assert payload["risk_example_backtest"]["hides_real_move_risk_count"] == 1
