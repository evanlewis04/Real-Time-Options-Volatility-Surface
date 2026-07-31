"""Project healthcheck for CI and local smoke testing.

Exercises the filings-intelligence keep-set offline: the RAG import surface, the
realized-volatility estimator, and an end-to-end unified analyst brief built from
the local cache with a deterministic market snapshot (no network, no OpenAI).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Deterministic offline market snapshot mirrored from the brief smoke: keeps the
# healthcheck fully local (no volatility engine, no yfinance, no OpenAI).
DETERMINISTIC_SNAPSHOT = {
    "source_mode": "Fallback",
    "message": "Deterministic offline market snapshot (not live).",
    "front_expected_move_pct": 8.2,
    "iv_rank": 64.0,
    "iv_30d": 0.52,
    "skew": -0.04,
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _run(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        detail = fn()
        return CheckResult(name, True, detail)
    except Exception as exc:
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}")


def check_imports() -> str:
    from src.financial_rag.api import build_local_api_service  # noqa: F401
    from src.financial_rag.differentiators import get_market_context  # noqa: F401
    from src.financial_rag.integration import (  # noqa: F401
        build_unified_brief,
        market_provider_from_metrics,
        volatility_market_provider,
    )
    from src.financial_rag.settings import project_root  # noqa: F401
    from src.marketdata.realized_vol import latest_realized_volatility  # noqa: F401

    return "financial_rag and marketdata modules imported"


def check_realized_vol() -> str:
    from src.marketdata.realized_vol import latest_realized_volatility, realized_volatility_estimators

    rng = np.random.default_rng(0)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=120)))
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": closes * 1.01,
            "Low": closes * 0.99,
            "Close": closes,
        }
    )
    estimates = realized_volatility_estimators(frame, windows=(20,))
    latest = latest_realized_volatility(estimates)
    close_to_close = latest.get("close_to_close_20d")
    if close_to_close is None or not np.isfinite(close_to_close):
        raise AssertionError(f"realized vol not finite: {latest}")
    return f"close_to_close_20d={close_to_close:.4f}"


def check_brief() -> str:
    from src.financial_rag.api import build_local_api_service
    from src.financial_rag.integration import build_unified_brief, market_provider_from_metrics
    from src.financial_rag.settings import project_root

    service = build_local_api_service(root=project_root(), use_voyage=False)
    provider = market_provider_from_metrics(DETERMINISTIC_SNAPSHOT)
    brief = build_unified_brief(
        service,
        question="How have NVIDIA data center demand disclosures changed over the last year?",
        ticker="NVDA",
        top_k=5,
        per_subquery_k=8,
        market_provider=provider,
        run_answer=False,
    )
    payload = brief.to_dict()

    if payload["filing_evidence"]["result_count"] < 1:
        raise AssertionError("brief returned no filing evidence")
    if not payload["filing_evidence"]["accepted_citations"]:
        raise AssertionError("brief returned no accepted citations")
    if payload["market_context"]["status"] != "ok":
        raise AssertionError(f"market context status={payload['market_context']['status']}")
    labels = {source["label"] for source in payload["data_sources"]}
    if len(labels) != 2:
        raise AssertionError(f"expected two labeled data sources, got {sorted(labels)}")
    return (
        f"results={payload['filing_evidence']['result_count']}, "
        f"citations={len(payload['filing_evidence']['accepted_citations'])}, "
        f"market={payload['market_context']['status']}"
    )


def main() -> int:
    checks: List[CheckResult] = [
        _run("imports", check_imports),
        _run("realized_vol", check_realized_vol),
        _run("brief", check_brief),
    ]

    print("PROJECT HEALTHCHECK")
    print("=" * 60)
    for result in checks:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status:4} {result.name:12} {result.detail}")

    failed = [result for result in checks if not result.passed]
    if failed:
        print("=" * 60)
        print(f"{len(failed)} check(s) failed")
        return 1
    print("=" * 60)
    print("All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
