"""Run deterministic fit-mode validation and backtesting on fixture snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fixtures.noisy_option_chain import (
    FIXTURE_NOW,
    FIXTURE_SPOT,
    checked_clean_chain,
    checked_noisy_chain,
    fixture_reason_buckets,
)
from src.quant.surface_validation import backtest_fit_modes, fixture_snapshot_record, validate_fit_modes


def fixture_validation_report() -> dict[str, Any]:
    """Return a reproducible validation/backtest report for local fixtures."""
    clean_chain, clean_no_arb = checked_clean_chain()
    noisy_chain, noisy_no_arb = checked_noisy_chain()
    clean_quality = float(clean_chain.attrs.get("data_quality_score") or 0.0)
    noisy_quality = float(noisy_chain.attrs.get("data_quality_score") or 0.0)
    clean_buckets = fixture_reason_buckets(clean_chain, clean_no_arb)
    noisy_buckets = fixture_reason_buckets(noisy_chain, noisy_no_arb)

    clean_snapshot = fixture_snapshot_record(
        "clean",
        clean_chain,
        FIXTURE_SPOT,
        FIXTURE_NOW,
        quality_score=clean_quality,
        reason_buckets=clean_buckets,
    )
    noisy_snapshot = fixture_snapshot_record(
        "noisy",
        noisy_chain,
        FIXTURE_SPOT,
        FIXTURE_NOW + pd.Timedelta(days=1),
        quality_score=noisy_quality,
        reason_buckets=noisy_buckets,
    )
    shifted_chain = _stable_quality_shift(clean_chain)
    shifted_snapshot = fixture_snapshot_record(
        "stable_quality_shift",
        shifted_chain,
        FIXTURE_SPOT,
        FIXTURE_NOW + pd.Timedelta(days=1),
        quality_score=clean_quality,
        reason_buckets=clean_buckets,
    )
    return {
        "description": "Deterministic fixture validation. All fitted, prior-assisted, and validation values are diagnostics, not market observations.",
        "validation": {
            "clean": validate_fit_modes(clean_chain, FIXTURE_SPOT),
            "noisy": validate_fit_modes(noisy_chain, FIXTURE_SPOT, baseline_chain=clean_chain),
        },
        "backtest": backtest_fit_modes([clean_snapshot, noisy_snapshot]),
        "risk_example_backtest": backtest_fit_modes([clean_snapshot, shifted_snapshot]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON instead of a compact table.")
    args = parser.parse_args()

    payload = fixture_validation_report()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_table(payload)


def _print_table(payload: dict[str, Any]) -> None:
    rows = []
    for fixture, validation in payload["validation"].items():
        for mode in validation.get("modes", []):
            rows.append(
                {
                    "fixture": fixture,
                    "mode": mode["mode"],
                    "available": mode.get("available"),
                    "oos_rmse": mode.get("oos_rmse"),
                    "no_arb_rate": mode.get("no_arbitrage_violation_rate"),
                    "p95_abs_residual": (mode.get("residual_quantiles") or {}).get("p95"),
                }
            )
    print(pd.DataFrame(rows).to_string(index=False))
    backtest = payload["backtest"]
    print(
        f"transitions={backtest.get('transition_count')} "
        f"robust_improvements={backtest.get('robust_improvement_count')} "
        f"hides_real_move_risks={backtest.get('hides_real_move_risk_count')}"
    )


def _stable_quality_shift(chain: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic stable-quality move that robust weighting can damp."""
    shifted = chain.copy()
    mask = (
        (pd.to_datetime(shifted["expiration"]).dt.strftime("%Y-%m-%d") == "2026-06-07")
        & pd.to_numeric(shifted["strike"], errors="coerce").between(185.0, 215.0)
    )
    shifted.loc[mask, "computedIV"] = pd.to_numeric(shifted.loc[mask, "computedIV"], errors="coerce") + 0.25
    shifted.loc[mask, "impliedVolatility"] = shifted.loc[mask, "computedIV"]
    shifted["fitWeight"] = 1.0
    shifted.loc[mask, "fitWeight"] = 0.001
    return shifted


if __name__ == "__main__":
    main()
