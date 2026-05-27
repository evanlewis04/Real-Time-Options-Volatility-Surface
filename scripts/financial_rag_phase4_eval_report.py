"""Write a tiny offline Phase 4 retrieval eval report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.evaluation import (
    PHASE3_ROUTED_RETRIEVAL_FIXTURES,
    build_retrieval_eval_report,
    write_retrieval_eval_report,
)
from src.financial_rag.settings import project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the Phase 4 offline retrieval eval report.")
    parser.add_argument("--k", type=int, default=5, help="Recall@k cutoff.")
    parser.add_argument(
        "--output",
        default="artifacts/rag_eval/phase4_retrieval_eval.json",
        help="Ignored output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_retrieval_eval_report(
        PHASE3_ROUTED_RETRIEVAL_FIXTURES,
        retrieved_by_query_id={},
        k=args.k,
    )
    output_path = project_root() / args.output
    write_retrieval_eval_report(report, output_path)
    print(f"Cases: {report['case_count']}")
    print(f"Labeled cases: {report['labeled_case_count']}")
    print(f"Unlabeled cases: {report['unlabeled_case_count']}")
    print(f"Recall@{report['k']}: {report['mean_recall_at_k']:.4f}")
    print(f"MRR: {report['mrr']:.4f}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
