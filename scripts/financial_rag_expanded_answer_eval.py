"""Run dry-run or opt-in live OpenAI answer evals over expanded fixtures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.api import LocalApiError, QueryRequest, build_local_api_service
from src.financial_rag.evaluation import (
    EXPANDED_ANSWER_CASES,
    apply_gold_labels_to_cases,
    build_answer_quality_report,
    build_retrieval_quality_report,
    filter_cases,
    gold_label_summary,
    resolve_gold_labels,
    write_csv_rows,
    write_json_report,
)
from src.financial_rag.retrieval import load_local_retrieval_corpus
from src.financial_rag.settings import configured_secret, load_environment, project_root
from src.financial_rag.synthesis import (
    DEFAULT_OPENAI_SYNTHESIS_MODEL,
    check_openai_readiness,
    synthesize_answer_from_query_payload,
)


def parse_args() -> argparse.Namespace:
    load_environment()
    parser = argparse.ArgumentParser(description="Run expanded answer-quality evals.")
    parser.add_argument("--tickers", default="", help="Comma-separated ticker filter.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases.")
    parser.add_argument("--top-k", type=int, default=5, help="Merged retrieval results.")
    parser.add_argument("--per-subquery-k", type=int, default=5, help="Per-subquery results.")
    parser.add_argument("--model", default=configured_secret("OPENAI_MODEL") or DEFAULT_OPENAI_SYNTHESIS_MODEL)
    parser.add_argument("--use-voyage", action="store_true", help="Use Voyage query embeddings when configured.")
    parser.add_argument("--live", action="store_true", help="Actually call OpenAI.")
    parser.add_argument(
        "--skip-retrieval-gate",
        action="store_true",
        help="Allow live OpenAI calls even when local retrieval thresholds are not met.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/rag_eval/expanded_answer_eval.json",
        help="Ignored JSON output path.",
    )
    parser.add_argument(
        "--csv-output",
        default="artifacts/rag_eval/expanded_answer_eval.csv",
        help="Ignored CSV output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    chunks, _embeddings = load_local_retrieval_corpus(root=project_root())
    gold_labels = resolve_gold_labels(chunks)
    labeled_cases = apply_gold_labels_to_cases(EXPANDED_ANSWER_CASES, gold_labels)
    cases = filter_cases(labeled_cases, tickers=tickers or None, max_cases=args.max_cases, answer_only=True)
    readiness = check_openai_readiness(model=args.model)
    if args.live and readiness.status != "ready":
        print("OpenAI live answer eval is not ready:")
        for issue in readiness.issues:
            print(f"- {issue}")
        return 2

    service = build_local_api_service(root=project_root(), use_voyage=args.use_voyage)
    answers: dict[str, dict[str, Any]] = {}
    retrieval_errors: dict[str, dict[str, Any]] = {}
    retrieval_payloads: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}: {case.question}")
        try:
            payload = service.query(
                QueryRequest(
                    question=case.question,
                    ticker=case.tickers[0],
                    top_k=args.top_k,
                    per_subquery_k=args.per_subquery_k,
                )
            )
        except LocalApiError as exc:
            retrieval_errors[case.case_id] = exc.to_dict()["error"]
            continue
        retrieval_payloads[case.case_id] = payload
        answer = synthesize_answer_from_query_payload(
            payload,
            question=case.question,
            model=args.model,
            dry_run=not args.live,
        )
        answers[case.case_id] = answer.to_dict()

    retrieval_report = build_retrieval_quality_report(cases, retrieval_payloads, k=args.top_k)
    retrieval_report["gold_labels"] = gold_label_summary(gold_labels)
    if args.live and not args.skip_retrieval_gate and not _retrieval_gate_passed(retrieval_report):
        print("OpenAI live answer eval blocked by retrieval quality gate.")
        print(f"Evidence-quality pass rate: {retrieval_report['evidence_quality_pass_rate']:.3f}")
        print(f"Gold Recall@{args.top_k}: {retrieval_report['mean_recall_at_k']:.3f}")
        print(f"Failure counts: {retrieval_report['failure_counts']}")
        return 3

    report = build_answer_quality_report(cases, answers)
    report["mode"] = "live" if args.live else "dry_run"
    report["model"] = args.model
    report["openai_readiness"] = readiness.to_dict()
    report["retrieval_errors"] = retrieval_errors
    report["retrieval_quality"] = {
        "section_source_hit_rate": retrieval_report["section_source_hit_rate"],
        "evidence_quality_pass_rate": retrieval_report["evidence_quality_pass_rate"],
        "gold_recall_at_k": retrieval_report["mean_recall_at_k"],
        "gold_mrr": retrieval_report["mrr"],
        "failure_counts": retrieval_report["failure_counts"],
        "gold_labels": retrieval_report["gold_labels"],
    }
    report["use_voyage"] = args.use_voyage
    json_path = write_json_report(report, project_root() / args.output)
    csv_rows = [
        {
            **result,
            "failures": "|".join(result.get("failures", [])),
        }
        for result in report["results"]
    ]
    csv_path = write_csv_rows(csv_rows, project_root() / args.csv_output)

    print(f"Expanded answer eval complete ({report['mode']})")
    print(f"Cases: {report['case_count']}")
    print(f"Companies: {', '.join(report['companies'])}")
    print(f"Pass rate: {report['pass_rate']:.3f}")
    print(f"Hallucinated citations: {report['hallucinated_citation_count']}")
    print(f"Uncited factual sentences: {report['uncited_sentence_count']}")
    print(f"Weak evidence cases: {report['weak_evidence_count']}")
    print(f"Evidence-quality pass rate: {retrieval_report['evidence_quality_pass_rate']:.3f}")
    print(f"Gold Recall@{args.top_k}: {retrieval_report['mean_recall_at_k']:.3f}")
    print(f"Retrieval errors: {len(retrieval_errors)}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    return 0


def _retrieval_gate_passed(report: dict[str, Any]) -> bool:
    failures = report.get("failure_counts", {})
    return (
        float(report.get("evidence_quality_pass_rate", 0.0)) >= 0.8
        and float(report.get("mean_recall_at_k", 0.0)) >= 0.6
        and int(failures.get("safe_harbor_only", 0)) == 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
