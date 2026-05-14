from datetime import datetime

from dashboard_connector import DashboardConnector
from src.dashboard.app_shell import (
    data_quality_actionability,
    fit_comparison_display_rows,
    fit_diagnostics_export_payload,
    fit_mode_state,
    quality_drop_alert_summary,
)
from src.data.models import MarketDataSnapshot
from src.data.snapshots import save_snapshot
from src.quant.provenance import CURRENT_FIT_PROVENANCE, CURRENT_ROBUST_FIT_PROVENANCE, ML_DENOISED_PROVENANCE


def test_fit_mode_state_labels_unavailable_estimate_modes_without_changing_truth():
    meta = {
        "surface_estimate_type": CURRENT_FIT_PROVENANCE,
        "surface_prior_applied": False,
        "fit_mode_comparison": [
            {"mode": "Robust SVI", "provenance": CURRENT_ROBUST_FIT_PROVENANCE},
            {"mode": "ML Denoised", "enabled": False, "provenance": ML_DENOISED_PROVENANCE},
        ],
    }

    prior = fit_mode_state("Prior Assisted", meta)
    ml = fit_mode_state("ML Denoised", meta)
    raw = fit_mode_state("Diagnostic Raw", {**meta, "svi_smiles": [{"expiration": "2026-06-19"}]})

    assert not prior["available"]
    assert "not applied" in prior["warning"]
    assert not ml["available"]
    assert "off by default" in ml["warning"]
    assert raw["estimate_type"] == "raw_quote_diagnostic_overlay"


def test_fit_comparison_display_rows_are_metadata_sourced():
    timestamp = datetime(2026, 5, 10, 12)
    meta = {
        "timestamp": timestamp,
        "fit_eligible_count": 40,
        "fit_excluded_count": 8,
        "no_arbitrage_violation_count": 3,
        "surface_prior_blend_weight": 0.2,
        "fit_mode_comparison": [
            {"mode": "Robust SVI", "status": "fitted", "weighted_rmse": 0.01, "rmse": 0.02},
            {"mode": "Prior Assisted", "status": "applied", "arbitrage_violations": 1, "prior_weight": 0.2},
        ],
    }

    rows = fit_comparison_display_rows(meta)

    assert rows[0]["fit_mode"] == "Robust SVI"
    assert rows[0]["eligible_rows"] == 40
    assert rows[0]["excluded_rows"] == 8
    assert rows[0]["unweighted_rmse"] == 0.02
    assert rows[1]["no_arb_violations"] == 1
    assert rows[1]["timestamp"] == "2026-05-10T12:00:00"


def test_data_quality_actionability_and_export_payload_include_phase7_fields():
    meta = {
        "surface_quality_score": 72.0,
        "quality_reason_buckets": {"wide_spread": 5},
        "fit_penalty_reason_buckets": {"stale_quote_penalty": 3},
        "fit_hard_rejection_reason_buckets": {"no_arbitrage_violation": 2},
        "no_arbitrage_violation_rows": 2,
        "no_arbitrage_excluded_count": 2,
        "expiry_quality": {
            "2026-06-19": {"score": 91.0, "surface_quotes": 10, "rejected_quotes": 1, "reason_buckets": {}},
            "2026-05-15": {"score": 62.0, "surface_quotes": 4, "rejected_quotes": 8, "reason_buckets": {"wide": 4}},
        },
        "fit_diagnostics": {
            "residual_diagnostics": {
                "top_residuals": [{"strike": 100.0, "residual": 0.05, "fit_weight": 0.2}]
            }
        },
        "svi_smiles": [
            {
                "expiration": "2026-06-19",
                "dte": 40.0,
                "residuals": [{"strike": 100.0, "fit_weight": 0.2, "residual": 0.05}],
            }
        ],
        "fit_mode_comparison": [{"mode": "Robust SVI"}],
        "post_fit_arbitrage": {"violation_count": 1},
    }

    action = data_quality_actionability(meta)
    export = fit_diagnostics_export_payload("AAPL", meta)

    assert action["top_penalty_reasons"][0]["reason"] == "wide_spread"
    assert action["worst_expiries"][0]["expiry"] == "2026-05-15"
    assert action["worst_residual_contracts"][0]["strike"] == 100.0
    assert action["suggested_preset"] == "Strict"
    assert export["symbol"] == "AAPL"
    assert export["row_weights"][0]["fit_weight"] == 0.2
    assert export["post_fit_arbitrage"]["violation_count"] == 1


def test_quality_drop_alert_compares_current_metadata_to_prior_snapshot(tmp_path):
    prior = MarketDataSnapshot(
        symbol="AAPL",
        spot=100.0,
        spot_timestamp=datetime(2026, 5, 9, 10),
        chain_timestamp=datetime(2026, 5, 9, 10),
        source="fixture",
        mode="Stored",
        quality_score=95.0,
        data_quality_score=95.0,
        quality_reason_buckets=(("wide_spread", 1),),
    )
    save_snapshot(prior, tmp_path)
    connector = DashboardConnector()
    connector.snapshot_dir = tmp_path

    result = connector._quality_drop_metadata(
        "AAPL",
        {
            "timestamp": datetime(2026, 5, 10, 10),
            "surface_quality_score": 78.0,
            "quality_reason_buckets": {"wide_spread": 5},
        },
    )["quality_drop_alert"]
    summary = quality_drop_alert_summary(result and {"quality_drop_alert": result})

    assert result["available"]
    assert result["triggered"]
    assert result["score_change"] == -17.0
    assert result["reason_bucket_delta"] == {"wide_spread": 4}
    assert summary["level"] == "warning"
