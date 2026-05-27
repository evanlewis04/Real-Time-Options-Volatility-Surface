"""Write a local Phase 5 differentiators report without SEC refetch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.api import build_local_api_service
from src.financial_rag.settings import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the Phase 5 local differentiators report.")
    parser.add_argument("--ticker", default="NVDA", help="Ticker to summarize.")
    parser.add_argument("--fact-name", default="Revenues", help="Local companyfacts fact name.")
    parser.add_argument(
        "--output",
        default="artifacts/rag_eval/phase5_differentiators_report.json",
        help="Ignored output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = build_local_api_service(root=project_root(), use_voyage=False)
    report = service.differentiators(ticker=args.ticker, fact_name=args.fact_name)
    output_path = project_root() / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Ticker: {report['ticker']}")
    print(f"Change records: {len(report['changes'])}")
    print(f"Language signal groups: {len(report['language_signals'])}")
    print(f"XBRL status: {report['xbrl']['status']}")
    print(f"Market context status: {report['market_context']['status']}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
