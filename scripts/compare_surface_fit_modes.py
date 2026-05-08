"""Compare current standard surface fit behavior on deterministic fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard_connector import DashboardConnector
from tests.fixtures.noisy_option_chain import (
    FIXTURE_SPOT,
    checked_clean_chain,
    checked_noisy_chain,
    clean_option_chain_raw,
    fixture_reason_buckets,
    noisy_option_chain_raw,
)


def fixture_fit_summary(name: str) -> dict[str, Any]:
    """Return deterministic standard-fit diagnostics for one fixture."""
    if name == "clean":
        raw = clean_option_chain_raw()
        checked, no_arb = checked_clean_chain()
    elif name == "noisy":
        raw = noisy_option_chain_raw()
        checked, no_arb = checked_noisy_chain()
    else:
        raise ValueError(f"Unknown fixture: {name}")

    surface_chain = DashboardConnector._surface_iv_chain(checked)
    fit_meta = DashboardConnector._svi_metadata(surface_chain, spot=FIXTURE_SPOT, iv_column="computedIV")
    residuals = _residuals(fit_meta.get("svi_smiles") or [])
    rejection_buckets = fixture_reason_buckets(checked, no_arb)
    normalized_rows = int(len(checked))
    rejected_rows = int(len(raw) - normalized_rows)

    return {
        "fixture": name,
        "mode": "standard_svi_current",
        "provenance": {
            "source": "deterministic_fixture",
            "observed_quote_input": "normalized yfinance-shaped rows",
            "iv_input": "fixture provider impliedVolatility copied to computedIV",
            "fit_policy": "current standard fit excluding no-arbitrage rows",
        },
        "raw_rows": int(len(raw)),
        "normalized_rows": normalized_rows,
        "fit_rows": int(len(surface_chain)),
        "rejected_rows": rejected_rows,
        "no_arbitrage_excluded_count": int(surface_chain.attrs.get("no_arbitrage_excluded_count", 0)),
        "quality_score": float(checked.attrs.get("data_quality_score") or 0.0),
        "reason_buckets": rejection_buckets,
        "fit_rmse": fit_meta.get("fit_diagnostics", {}).get("rmse"),
        "fit_mae": fit_meta.get("fit_diagnostics", {}).get("mae"),
        "fit_max_error": fit_meta.get("fit_diagnostics", {}).get("max_error"),
        "fit_expiries": fit_meta.get("fit_diagnostics", {}).get("fitted_expiries"),
        "residual_quantiles": _quantiles(residuals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit deterministic JSON instead of a compact table.")
    args = parser.parse_args()

    payload = {
        "description": "Current standard SVI fixture comparison. Robust and ML-denoised modes are not market truth.",
        "fixtures": [fixture_fit_summary("clean"), fixture_fit_summary("noisy")],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_table(payload["fixtures"])


def _residuals(svi_smiles: list[dict[str, Any]]) -> list[float]:
    residuals: list[float] = []
    for smile in svi_smiles:
        for row in smile.get("residuals") or []:
            residual = row.get("residual")
            if residual is not None and np.isfinite(float(residual)):
                residuals.append(float(residual))
    return residuals


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "p99": None}
    series = pd.Series(values, dtype="float64")
    return {name: float(series.quantile(q)) for name, q in {"p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99}.items()}


def _print_table(rows: list[dict[str, Any]]) -> None:
    table = pd.DataFrame(
        [
            {
                "fixture": row["fixture"],
                "quality": row["quality_score"],
                "raw": row["raw_rows"],
                "normalized": row["normalized_rows"],
                "fit_rows": row["fit_rows"],
                "excluded": row["no_arbitrage_excluded_count"],
                "rmse": row["fit_rmse"],
                "p95_residual": row["residual_quantiles"]["p95"],
            }
            for row in rows
        ]
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
