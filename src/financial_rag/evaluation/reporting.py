"""Small offline retrieval report helpers for Phase 4."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from src.financial_rag.evaluation.retrieval_eval import (
    RetrievalEvalCase,
    evaluate_retrieval_results,
    mean_reciprocal_rank,
)


def build_retrieval_eval_report(
    cases: Sequence[RetrievalEvalCase],
    retrieved_by_query_id: dict[str, Sequence[str]],
    *,
    k: int = 5,
) -> dict[str, object]:
    """Build an offline report, marking unlabeled fixtures honestly."""

    labeled = [case for case in cases if case.relevant_chunk_ids]
    unlabeled = [case for case in cases if not case.relevant_chunk_ids]
    results = evaluate_retrieval_results(labeled, retrieved_by_query_id, k=k)
    return {
        "k": k,
        "case_count": len(cases),
        "labeled_case_count": len(labeled),
        "unlabeled_case_count": len(unlabeled),
        "mean_recall_at_k": _mean([result.recall_at_k for result in results]),
        "mrr": mean_reciprocal_rank([result.reciprocal_rank for result in results]),
        "results": [asdict(result) for result in results],
        "unlabeled_queries": [
            {
                "query_id": case.query_id,
                "question": case.question,
                "status": "unlabeled",
                "expected_query_type": case.expected_query_type,
            }
            for case in unlabeled
        ],
    }


def write_retrieval_eval_report(report: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
