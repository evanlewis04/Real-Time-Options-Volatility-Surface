import numpy as np

from dashboard_connector import DashboardConnector
from src.quant.surface_arbitrage import (
    DIAGNOSTIC_PROVENANCE,
    REPAIR_PROVENANCE,
    check_surface_arbitrage,
    conservative_surface_repair,
    surface_comparison_rows,
)
from tests.fixtures.noisy_option_chain import FIXTURE_SPOT, checked_clean_chain


def test_surface_arbitrage_checks_identify_calendar_and_convexity_violations():
    strikes = np.array([90.0, 100.0, 110.0])
    expiries = np.array([30.0, 60.0, 90.0])
    vols = np.array(
        [
            [0.20, 0.80, 0.20],
            [0.21, 0.18, 0.21],
            [0.22, 0.95, -0.10],
        ]
    )
    chain, _ = checked_clean_chain()

    diagnostics = check_surface_arbitrage(
        strikes,
        expiries,
        vols,
        FIXTURE_SPOT,
        input_rows=chain,
        surface_label="fixture_surface",
    )

    assert not diagnostics["passed"]
    assert diagnostics["provenance"] == DIAGNOSTIC_PROVENANCE
    assert diagnostics["reason_buckets"]["calendar_monotonicity"] >= 1
    assert diagnostics["reason_buckets"]["butterfly_convexity"] >= 1
    assert diagnostics["reason_buckets"]["positive_vol"] == 1
    assert diagnostics["suggestions"][0]["likely_input_rows"]


def test_conservative_surface_repair_reduces_violations_and_labels_estimates():
    strikes = np.array([90.0, 100.0, 110.0])
    expiries = np.array([30.0, 60.0, 90.0])
    vols = np.array(
        [
            [0.20, 0.80, 0.20],
            [0.21, 0.18, 0.21],
            [0.22, 0.95, 0.22],
        ]
    )

    disabled, disabled_meta = conservative_surface_repair(strikes, expiries, vols, FIXTURE_SPOT, enabled=False)
    repaired, meta = conservative_surface_repair(strikes, expiries, vols, FIXTURE_SPOT, enabled=True)

    assert np.array_equal(disabled, vols)
    assert not disabled_meta["applied"]
    assert meta["provenance"] == REPAIR_PROVENANCE
    assert meta["applied"]
    assert meta["after_violation_count"] < meta["before_violation_count"]
    assert meta["repaired_cell_count"] > 0
    assert meta["repair_records"][0]["provenance"] == REPAIR_PROVENANCE
    assert not np.array_equal(repaired, vols)


def test_surface_comparison_rows_are_deterministic_for_prior_and_repair_modes():
    strikes = np.array([90.0, 100.0, 110.0])
    expiries = np.array([30.0, 60.0])
    current = np.array([[0.25, 0.24, 0.25], [0.27, 0.26, 0.27]])
    prior = current + 0.01
    repaired, repair_meta = conservative_surface_repair(strikes, expiries, prior, FIXTURE_SPOT, enabled=True)

    rows = surface_comparison_rows(
        strikes,
        expiries,
        FIXTURE_SPOT,
        current_vols=current,
        prior_assisted_vols=prior,
        repaired_vols=repaired,
        prior_metadata={"applied": True, "blend_weight": 0.20},
        repair_metadata=repair_meta,
    )

    assert [row["mode"] for row in rows] == ["Robust Surface", "Prior Assisted", "Conservative Repair"]
    assert rows[1]["prior_weight"] == 0.20
    assert rows[2]["status"] == "candidate"
    assert all("arbitrage_violations" in row for row in rows)


def test_connector_fit_comparison_keeps_ml_off_and_adds_surface_diagnostics():
    base = [
        {"mode": "Standard SVI", "rmse": 0.02},
        {"mode": "Robust SVI", "rmse": 0.01},
        {"mode": "ML Denoised", "status": "research_off_by_default"},
    ]
    surface_rows = [
        {
            "mode": "Prior Assisted",
            "arbitrage_violations": 0,
            "prior_weight": 0.15,
        }
    ]

    rows = DashboardConnector._append_surface_comparison_rows(base, surface_rows)

    assert [row["mode"] for row in rows] == ["Standard SVI", "Robust SVI", "ML Denoised", "Prior Assisted"]
    assert rows[0]["surface_role"] == "raw_standard_fit"
    assert rows[2]["surface_role"] == "ml_research_off"
    assert rows[3]["prior_weight"] == 0.15
