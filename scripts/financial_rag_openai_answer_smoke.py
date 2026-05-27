"""Prepare or run an opt-in OpenAI answer smoke over local retrieved evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.api import QueryRequest, build_local_api_service
from src.financial_rag.audit import build_evidence_quality_report, write_json_report
from src.financial_rag.settings import configured_secret, load_environment, project_root
from src.financial_rag.synthesis import (
    DEFAULT_OPENAI_SYNTHESIS_MODEL,
    check_openai_readiness,
    synthesize_answer_from_query_payload,
)


DEFAULT_QUERY = "What risks does NVIDIA describe?"


def parse_args() -> argparse.Namespace:
    load_environment()
    default_model = configured_secret("OPENAI_MODEL") or DEFAULT_OPENAI_SYNTHESIS_MODEL
    parser = argparse.ArgumentParser(description="Run an OpenAI-ready answer smoke over local filings evidence.")
    parser.add_argument("--ticker", default="NVDA", help="Ticker to inspect. Default: NVDA.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Question to retrieve and answer.")
    parser.add_argument("--model", default=default_model, help="OpenAI model for live mode.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieved evidence count.")
    parser.add_argument("--per-subquery-k", type=int, default=5, help="Per-subquery retrieval count.")
    parser.add_argument(
        "--use-voyage",
        action="store_true",
        help="Use Voyage query embeddings for retrieval when VOYAGE_API_KEY is configured.",
    )
    parser.add_argument("--live", action="store_true", help="Actually call OpenAI. Omit for dry-run readiness.")
    parser.add_argument(
        "--output",
        default="artifacts/rag_eval/openai_answer_smoke.json",
        help="Ignored JSON output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    service = build_local_api_service(root=project_root(), use_voyage=args.use_voyage)
    query_payload = service.query(
        QueryRequest(
            question=args.query,
            ticker=args.ticker,
            top_k=args.top_k,
            per_subquery_k=args.per_subquery_k,
        )
    )
    evidence_quality = build_evidence_quality_report(query_payload)
    readiness = check_openai_readiness(model=args.model)
    if args.live and readiness.status != "ready":
        print("OpenAI live smoke is not ready:")
        for issue in readiness.issues:
            print(f"- {issue}")
        return 2

    answer = synthesize_answer_from_query_payload(
        query_payload,
        question=args.query,
        model=args.model,
        dry_run=not args.live,
    )
    report = {
        "mode": "live" if args.live else "dry_run",
        "retrieval": {"use_voyage": args.use_voyage},
        "openai_readiness": readiness.to_dict(),
        "evidence_quality": evidence_quality.to_dict(),
        "answer": answer.to_dict(),
    }
    output_path = write_json_report(report, project_root() / args.output)

    print(f"OpenAI answer smoke: {answer.status.upper()} ({report['mode']})")
    print(f"Model: {args.model}")
    print(f"OpenAI readiness: {readiness.status}")
    for issue in readiness.issues:
        print(f"- {issue}")
    print(f"Evidence quality: {evidence_quality.status}, results={evidence_quality.result_count}")
    print(f"Accepted citations: {len(answer.accepted_citations)}")
    print(f"Rejected citations: {len(answer.rejected_citations)}")
    print(f"Output: {output_path}")
    return 0 if answer.status in {"pass", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
