"""Tiny offline retrieval eval scaffolding for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class RetrievalEvalCase:
    query_id: str
    question: str
    relevant_chunk_ids: set[str] = field(default_factory=set)
    expected_ticker: str = "NVDA"
    expected_query_type: str = "single_doc_lookup"
    notes: str = ""


@dataclass(frozen=True)
class RetrievalEvalResult:
    query_id: str
    recall_at_k: float
    reciprocal_rank: float


NVDA_RETRIEVAL_FIXTURES: tuple[RetrievalEvalCase, ...] = (
    RetrievalEvalCase("nvda-risk-factors", "What risks does NVIDIA describe?", notes="10-K risk factors."),
    RetrievalEvalCase("nvda-data-center", "What does NVIDIA say about data center demand?"),
    RetrievalEvalCase("nvda-gross-margin", "What changed in NVIDIA gross margin commentary?"),
    RetrievalEvalCase("nvda-supply", "How does NVIDIA describe supply constraints or capacity?"),
    RetrievalEvalCase("nvda-export-controls", "What does NVIDIA disclose about export controls?"),
    RetrievalEvalCase("nvda-cfo-commentary", "What did NVIDIA's CFO commentary say about revenue?"),
    RetrievalEvalCase("nvda-gaming", "What does NVIDIA report about gaming revenue?"),
)

PHASE3_ROUTED_RETRIEVAL_FIXTURES: tuple[RetrievalEvalCase, ...] = (
    RetrievalEvalCase(
        "phase3-temporal-risk",
        "How have NVIDIA risk disclosures changed over the last year?",
        expected_query_type="temporal",
    ),
    RetrievalEvalCase(
        "phase3-cross-company-data-center",
        "Compare NVDA and AMD data center commentary.",
        expected_query_type="cross_company",
    ),
    RetrievalEvalCase(
        "phase3-cfo-commentary",
        "What did NVIDIA CFO commentary say about revenue?",
        expected_query_type="speaker_specific",
    ),
    RetrievalEvalCase(
        "phase3-item-risk",
        "What does NVDA Item 1A say about export controls?",
        expected_query_type="single_doc_lookup",
    ),
    RetrievalEvalCase(
        "phase3-cross-source",
        "Does NVIDIA 10-Q risk language match CFO commentary?",
        expected_query_type="cross_source",
    ),
)


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
    *,
    k: int,
) -> float:
    """Compute single-query Recall@k."""

    if not relevant_chunk_ids:
        return 0.0
    retrieved = set(retrieved_chunk_ids[:k])
    return len(retrieved & relevant_chunk_ids) / len(relevant_chunk_ids)


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: set[str],
) -> float:
    """Return reciprocal rank for the first relevant retrieved chunk."""

    if not relevant_chunk_ids:
        return 0.0
    for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(ranks: Sequence[float]) -> float:
    if not ranks:
        return 0.0
    return sum(ranks) / len(ranks)


def evaluate_retrieval_results(
    cases: Sequence[RetrievalEvalCase],
    retrieved_by_query_id: dict[str, Sequence[str]],
    *,
    k: int = 5,
) -> list[RetrievalEvalResult]:
    """Evaluate already-run retrieval results without any network or LLM calls."""

    results: list[RetrievalEvalResult] = []
    for case in cases:
        retrieved = retrieved_by_query_id.get(case.query_id, ())
        results.append(
            RetrievalEvalResult(
                query_id=case.query_id,
                recall_at_k=recall_at_k(retrieved, case.relevant_chunk_ids, k=k),
                reciprocal_rank=reciprocal_rank(retrieved, case.relevant_chunk_ids),
            )
        )
    return results
