"""Run the Phase 7 local API/service smoke checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.api import DEFAULT_PHASE7_QUERY, run_api_smoke
from src.financial_rag.settings import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Phase 7 API smoke checks.")
    parser.add_argument("--ticker", default="NVDA", help="Ticker to inspect. Default: NVDA.")
    parser.add_argument("--query", default=DEFAULT_PHASE7_QUERY, help="Query smoke question.")
    parser.add_argument(
        "--output",
        default="artifacts/rag_eval/phase7_api_smoke.json",
        help="Ignored JSON output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = project_root() / args.output
    report = run_api_smoke(
        root=project_root(),
        ticker=args.ticker,
        query=args.query,
        output_path=output_path,
    )
    print(f"Phase 7 API smoke: {report.status.upper()}")
    print(f"Endpoint contracts: {report.endpoint_count}")
    for step in report.steps:
        print(f"- {step.status.upper()} {step.name}: {step.message}")
        if step.details:
            details = ", ".join(f"{key}={value}" for key, value in step.details.items())
            print(f"  {details}")
    print(f"Output: {report.artifact_path}")
    return 0 if report.status in {"pass", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
