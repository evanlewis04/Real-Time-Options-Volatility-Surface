"""Run the Phase 6 local recruiter-demo workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.demo import DEFAULT_PHASE6_QUERY, run_demo_workflow
from src.financial_rag.settings import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 6 local filings demo workflow.")
    parser.add_argument("--ticker", default="NVDA", help="Ticker to inspect; defaults to NVDA.")
    parser.add_argument("--query", default=DEFAULT_PHASE6_QUERY, help="Query smoke question.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Artifact directory; defaults to artifacts/rag_eval.",
    )
    args = parser.parse_args()

    report = run_demo_workflow(
        root=project_root(),
        ticker=args.ticker,
        query=args.query,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Phase 6 demo workflow: {report.status.upper()}")
    print("Prerequisites:")
    for prerequisite in report.prerequisites:
        print(f"- {prerequisite}")
    print("Steps:")
    for step in report.steps:
        print(f"- {step.status.upper()} {step.name}: {step.message}")
        if step.details:
            details = ", ".join(f"{key}={value}" for key, value in step.details.items())
            print(f"  {details}")
    print("Artifacts:")
    for name, path in report.artifact_paths.items():
        print(f"- {name}: {path}")
    print("Next actions:")
    for action in report.next_actions:
        print(f"- {action}")


if __name__ == "__main__":
    main()
