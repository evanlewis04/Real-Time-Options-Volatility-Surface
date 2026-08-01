"""Phase 3 local query pipeline over deterministic routing and dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.financial_rag.query.coverage import CoverageReport, build_coverage_report
from src.financial_rag.query.parent_context import HydratedContext, hydrate_parent_context
from src.financial_rag.query.planning import RetrievalSubquery, plan_retrieval
from src.financial_rag.query.router import RoutedQuery, route_query
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever, RetrievalResult
from src.financial_rag.retrieval.rerank import RerankCandidate, Reranker
from src.financial_rag.synthesis import CitationValidation, validate_citations


@dataclass(frozen=True)
class QueryRetrievalResult:
    chunk_id: str
    rank: int
    dense_score: float
    citation_label: str
    source_url: str
    source_excerpt: str
    metadata: dict[str, object]
    subquery_id: str
    trace: dict[str, object] = field(default_factory=dict)
    parent_context: HydratedContext | None = None


@dataclass(frozen=True)
class QueryPipelineResult:
    routed_query: RoutedQuery
    subqueries: list[RetrievalSubquery]
    results: list[QueryRetrievalResult]
    citation_validation: CitationValidation
    coverage: CoverageReport
    trace: dict[str, object] = field(default_factory=dict)


class QueryPipeline:
    """Run route -> plan -> retrieve -> hydrate -> validate over local data."""

    def __init__(
        self,
        *,
        retriever: LocalDenseRetriever,
        chunks: list[LocalChunkRecord],
        reranker: Reranker | None = None,
        candidate_k: int = 30,
    ) -> None:
        self.retriever = retriever
        self.chunks = chunks
        self.reranker = reranker
        self.candidate_k = max(candidate_k, 1)
        self._chunk_text_by_id = {chunk.chunk_id: chunk.chunk_text for chunk in chunks}

    def run(
        self,
        question: str,
        *,
        default_ticker: str | None = None,
        top_k: int = 5,
        per_subquery_k: int | None = None,
        as_of: datetime | None = None,
    ) -> QueryPipelineResult:
        routed = route_query(question, default_ticker=default_ticker)
        subqueries = plan_retrieval(routed)
        base_k = per_subquery_k or top_k
        search_k = max(self.candidate_k, base_k) if self.reranker is not None else base_k
        raw_results: list[tuple[str, RetrievalResult, dict[str, object]]] = []
        for subquery in subqueries:
            for result in self.retriever.search(
                query=subquery.query,
                top_k=search_k,
                filters=subquery.filters,
                as_of=as_of,
            ):
                raw_results.append((subquery.subquery_id, result, dict(subquery.trace)))

        merged = _merge_results(
            raw_results,
            chunks=self.chunks,
            top_k=top_k,
            question=question,
            reranker=self.reranker,
            chunk_text_by_id=self._chunk_text_by_id,
        )
        citation_validation = validate_citations([result.citation_label for result in merged], _as_retrieval_results(merged))
        coverage = build_coverage_report(self.chunks, tickers=routed.filters.tickers)
        return QueryPipelineResult(
            routed_query=routed,
            subqueries=subqueries,
            results=merged,
            citation_validation=citation_validation,
            coverage=coverage,
            trace={
                "pipeline": "phase3_local_query_v1",
                "subquery_count": len(subqueries),
                "result_count": len(merged),
                "as_of": as_of.isoformat() if as_of is not None else None,
            },
        )


def _merge_results(
    raw_results: list[tuple[str, RetrievalResult, dict[str, object]]],
    *,
    chunks: list[LocalChunkRecord],
    top_k: int,
    question: str = "",
    reranker: Reranker | None = None,
    chunk_text_by_id: dict[str, str] | None = None,
) -> list[QueryRetrievalResult]:
    chunk_text_by_id = chunk_text_by_id or {}
    best_by_chunk: dict[str, tuple[str, RetrievalResult, dict[str, object]]] = {}
    for subquery_id, result, trace in raw_results:
        current = best_by_chunk.get(result.chunk_id)
        if current is None or _result_score(result) > _result_score(current[1]):
            best_by_chunk[result.chunk_id] = (subquery_id, result, trace)

    candidates = list(best_by_chunk.values())
    rerank_scores: dict[str, float] = {}
    combined_scores: dict[str, float] = {}
    if reranker is not None and candidates:
        rerank_inputs = [
            RerankCandidate(
                chunk_id=item[1].chunk_id,
                text=chunk_text_by_id.get(item[1].chunk_id, item[1].source_excerpt),
                metadata=dict(item[1].metadata),
            )
            for item in candidates
        ]
        scores = reranker.score(question, rerank_inputs)
        rerank_scores = {candidate.chunk_id: float(score) for candidate, score in zip(rerank_inputs, scores, strict=True)}
        # Reciprocal Rank Fusion of the first-stage and rerank orderings. RRF is
        # rank-based and scale-free, so the reranker refines ordering without
        # discarding the dense, metadata-intent, and safe-harbor signals the
        # first stage encodes: a chunk the first stage ranks highly stays near
        # the top even if the reranker scores it lower.
        first_stage = {item[1].chunk_id: _result_score(item[1]) for item in candidates}
        first_rank = _rank_positions(first_stage)
        rerank_rank = _rank_positions(rerank_scores)
        fallback_rank = len(candidates) + 1
        combined_scores = {
            chunk_id: 1.0 / (_RRF_C + first_rank[chunk_id])
            + _RERANK_BLEND_WEIGHT / (_RRF_C + rerank_rank.get(chunk_id, fallback_rank))
            for chunk_id in first_stage
        }

    # First-stage ordering selects which chunks are surfaced (recall and
    # source-hit are determined here). Diversify across subqueries, then take the
    # final set. The reranker only reorders that set, so it can lift MRR/NDCG
    # without ever demoting a source-satisfying chunk out of the final top-k.
    first_stage_ordered = sorted(candidates, key=lambda item: _result_score(item[1]), reverse=True)
    diversified: list[tuple[str, RetrievalResult, dict[str, object]]] = []
    seen_subqueries: set[str] = set()
    for item in first_stage_ordered:
        subquery_id = item[0]
        if subquery_id in seen_subqueries:
            continue
        diversified.append(item)
        seen_subqueries.add(subquery_id)
    diversified_ids = {item[1].chunk_id for item in diversified}
    diversified.extend(item for item in first_stage_ordered if item[1].chunk_id not in diversified_ids)

    selected = diversified[:top_k]
    if reranker is not None and selected:
        selected = sorted(selected, key=lambda item: combined_scores.get(item[1].chunk_id, 0.0), reverse=True)

    merged: list[QueryRetrievalResult] = []
    for rank, (subquery_id, result, trace) in enumerate(selected, start=1):
        label = f"S{rank}"
        parent = hydrate_parent_context(result, chunks)
        metadata = dict(result.metadata)
        metadata["citation_label"] = label
        merged_trace = dict(trace)
        if reranker is not None:
            merged_trace["reranker"] = reranker.name
            merged_trace["rerank_score"] = rerank_scores.get(result.chunk_id, 0.0)
        merged.append(
            QueryRetrievalResult(
                chunk_id=result.chunk_id,
                rank=rank,
                dense_score=_result_score(result),
                citation_label=label,
                source_url=result.source_url,
                source_excerpt=result.source_excerpt,
                metadata=metadata,
                subquery_id=subquery_id,
                trace=merged_trace,
                parent_context=parent,
            )
        )
    return merged


def _as_retrieval_results(results: list[QueryRetrievalResult]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=result.chunk_id,
            rank=result.rank,
            dense_score=result.dense_score,
            citation_label=result.citation_label,
            source_url=result.source_url,
            source_excerpt=result.source_excerpt,
            metadata=dict(result.metadata),
        )
        for result in results
    ]


def _result_score(result: RetrievalResult) -> float:
    return result.score or result.dense_score


# RRF constant and the weight of the rerank ordering relative to the first-stage
# ordering. Standard RRF uses C=60; the first-stage term is unweighted so it
# stays dominant and source-satisfying first-stage results are not demoted out
# of the final top-k.
_RRF_C = 60.0
_RERANK_BLEND_WEIGHT = 1.0


def _rank_positions(scores: dict[str, float]) -> dict[str, int]:
    """Map each id to its 1-based rank by descending score (stable on ties)."""

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {chunk_id: index for index, (chunk_id, _score) in enumerate(ordered, start=1)}
