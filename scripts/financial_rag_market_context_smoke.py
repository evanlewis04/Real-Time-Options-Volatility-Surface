"""Smoke for the thin RAG-evidence + market-context integration prototype.

Builds a combined brief for one question/ticker: cited filing evidence plus
market-implied context, with explicit data-source labels. Cache-only by default
(a deterministic offline market snapshot); pass --live-market to use the
volatility engine. Writes an ignored artifact under artifacts/rag_eval/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.api import build_local_api_service
from src.financial_rag.integration import (
    build_brief_from_service,
    market_provider_from_metrics,
    volatility_market_provider,
)
from src.financial_rag.settings import project_root


DETERMINISTIC_SNAPSHOT = {
    "source_mode": "Fallback",
    "message": "Deterministic offline market snapshot (not live).",
    "front_expected_move_pct": 8.2,
    "iv_rank": 64.0,
    "iv_30d": 0.52,
    "skew": -0.04,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG evidence + market context brief smoke.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument(
        "--question",
        default="How have NVIDIA data center demand disclosures changed over the last year?",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--per-subquery-k", type=int, default=8)
    parser.add_argument("--use-voyage", action="store_true")
    parser.add_argument(
        "--live-market",
        action="store_true",
        help="Use the volatility engine provider instead of the deterministic snapshot.",
    )
    parser.add_argument("--output", default="artifacts/rag_eval/market_context_brief.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = build_local_api_service(root=project_root(), use_voyage=args.use_voyage)
    provider = volatility_market_provider if args.live_market else market_provider_from_metrics(DETERMINISTIC_SNAPSHOT)
    brief = build_brief_from_service(
        service,
        question=args.question,
        ticker=args.ticker,
        top_k=args.top_k,
        per_subquery_k=args.per_subquery_k,
        market_provider=provider,
    )
    payload = brief.to_dict()
    output_path = project_root() / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Ticker: {brief.ticker}")
    print(f"Filing evidence results: {payload['filing_evidence']['result_count']}")
    print(f"Accepted citations: {len(payload['filing_evidence']['accepted_citations'])}")
    print(
        "Market context: "
        f"status={payload['market_context']['status']} "
        f"source_mode={payload['market_context']['source_mode']}"
    )
    print(f"Market metrics: {payload['market_context']['metrics']}")
    print("Data sources:")
    for source in payload["data_sources"]:
        print(f"  - {source['label']} ({source['kind']}): {source['provenance']}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
