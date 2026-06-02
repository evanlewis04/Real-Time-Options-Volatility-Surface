"""Smoke for the unified analyst brief: cited filing evidence + optional gated
answer + labeled market context for one question/ticker.

Cache-only by default (a deterministic offline market snapshot, no OpenAI call);
pass --live-market to source market metrics from the volatility engine and
--answer to attempt the gated OpenAI answer. Writes an ignored artifact.
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
    build_unified_brief,
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
    parser = argparse.ArgumentParser(description="Unified analyst brief smoke.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument(
        "--question",
        default="How have NVIDIA data center demand disclosures changed over the last year?",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--per-subquery-k", type=int, default=8)
    parser.add_argument("--use-voyage", action="store_true")
    parser.add_argument("--live-market", action="store_true", help="Use the volatility engine for market metrics.")
    parser.add_argument("--answer", action="store_true", help="Attempt the gated OpenAI answer (opt-in).")
    parser.add_argument("--output", default="artifacts/rag_eval/unified_brief.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = build_local_api_service(root=project_root(), use_voyage=args.use_voyage)
    provider = volatility_market_provider if args.live_market else market_provider_from_metrics(DETERMINISTIC_SNAPSHOT)
    brief = build_unified_brief(
        service,
        question=args.question,
        ticker=args.ticker,
        top_k=args.top_k,
        per_subquery_k=args.per_subquery_k,
        market_provider=provider,
        run_answer=args.answer,
    )
    payload = brief.to_dict()
    output_path = project_root() / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Ticker: {brief.ticker}")
    print(f"Filing evidence results: {payload['filing_evidence']['result_count']}")
    print(f"Accepted citations: {len(payload['filing_evidence']['accepted_citations'])}")
    answer = payload["answer"]
    if answer is None:
        print(f"Answer: not generated (gate allowed={payload['answer_gate']['allowed']})")
        for reason in payload["answer_gate"]["reasons"]:
            print(f"  - {reason}")
    else:
        print(f"Answer: {answer['answer_text']}")
    print(
        "Market context: "
        f"status={payload['market_context']['status']} "
        f"source_mode={payload['market_context']['source_mode']}"
    )
    print("Data sources:")
    for source in payload["data_sources"]:
        print(f"  - {source['label']} ({source['kind']}): {source['provenance']}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
