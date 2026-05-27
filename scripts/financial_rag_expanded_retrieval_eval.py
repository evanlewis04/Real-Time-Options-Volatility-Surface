"""Run expanded offline retrieval evals across local and fixture companies."""

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
    EXPANDED_RETRIEVAL_CASES,
    apply_gold_labels_to_cases,
    build_retrieval_quality_report,
    filter_cases,
    gold_label_summary,
    resolve_gold_labels,
    write_csv_rows,
    write_json_report,
)
from src.financial_rag.retrieval import load_local_retrieval_corpus
from src.financial_rag.settings import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run expanded local retrieval-quality evals.")
    parser.add_argument("--tickers", default="", help="Comma-separated ticker filter.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit cases.")
    parser.add_argument("--top-k", type=int, default=5, help="Merged retrieval results.")
    parser.add_argument("--per-subquery-k", type=int, default=5, help="Per-subquery results.")
    parser.add_argument("--use-voyage", action="store_true", help="Use Voyage query embeddings when configured.")
    parser.add_argument(
        "--output",
        default="artifacts/rag_eval/expanded_retrieval_eval.json",
        help="Ignored JSON output path.",
    )
    parser.add_argument(
        "--csv-output",
        default="artifacts/rag_eval/expanded_retrieval_eval.csv",
        help="Ignored CSV output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    chunks, _embeddings = load_local_retrieval_corpus(root=project_root())
    gold_labels = resolve_gold_labels(chunks)
    labeled_cases = apply_gold_labels_to_cases(EXPANDED_RETRIEVAL_CASES, gold_labels)
    cases = filter_cases(labeled_cases, tickers=tickers or None, max_cases=args.max_cases)
    service = build_local_api_service(root=project_root(), use_voyage=args.use_voyage)
    payloads: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}: {case.question}")
        try:
            payloads[case.case_id] = service.query(
                QueryRequest(
                    question=case.question,
                    ticker=case.tickers[0],
                    top_k=args.top_k,
                    per_subquery_k=args.per_subquery_k,
                )
            )
        except LocalApiError as exc:
            payloads[case.case_id] = exc.to_dict()

    report = build_retrieval_quality_report(cases, payloads, k=args.top_k)
    report["gold_labels"] = gold_label_summary(gold_labels)
    report["gold_label_specs"] = len(gold_labels)
    coverage_tickers = sorted(
        {ticker for case in cases for ticker in case.tickers if ticker in {"NVDA", "AMD", "MSFT", "AAPL", "JPM", "XOM"}}
    )
    report["coverage"] = service.coverage(tickers=coverage_tickers)
    json_path = write_json_report(report, project_root() / args.output)
    csv_rows = [
        {
            **result,
            "failures": "|".join(result.get("failures", [])),
        }
        for result in report["results"]
    ]
    csv_path = write_csv_rows(csv_rows, project_root() / args.csv_output)

    print("Expanded retrieval eval complete")
    print(f"Cases: {report['case_count']}")
    print(f"Companies: {', '.join(report['companies'])}")
    print(f"Section/source hit rate: {report['section_source_hit_rate']:.3f}")
    print(f"Metadata completeness rate: {report['metadata_completeness_rate']:.3f}")
    print(f"Evidence-quality pass rate: {report['evidence_quality_pass_rate']:.3f}")
    print(f"Gold labels resolved: {report['gold_labels']['label_count']}")
    print(f"Gold Recall@{args.top_k}: {report['mean_recall_at_k']:.3f}")
    print(f"Gold MRR: {report['mrr']:.3f}")
    print(f"Failure counts: {report['failure_counts']}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
