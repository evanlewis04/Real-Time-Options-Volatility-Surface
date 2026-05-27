"""Phase 3 local query pipeline over deterministic routing and dense retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.financial_rag.query.coverage import CoverageReport, build_coverage_report
from src.financial_rag.query.parent_context import HydratedContext, hydrate_parent_context
from src.financial_rag.query.planning import RetrievalSubquery, plan_retrieval
from src.financial_rag.query.router import RoutedQuery, route_query
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever, RetrievalResult
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
    ) -> None:
        self.retriever = retriever
        self.chunks = chunks

    def run(
        self,
        question: str,
        *,
        default_ticker: str | None = None,
        top_k: int = 5,
        per_subquery_k: int | None = None,
    ) -> QueryPipelineResult:
        routed = route_query(question, default_ticker=default_ticker)
        subqueries = plan_retrieval(routed)
        raw_results: list[tuple[str, RetrievalResult, dict[str, object]]] = []
        for subquery in subqueries:
            for result in self.retriever.search(
                query=subquery.query,
                top_k=per_subquery_k or top_k,
                filters=subquery.filters,
            ):
                raw_results.append((subquery.subquery_id, result, dict(subquery.trace)))

        merged = _merge_results(raw_results, chunks=self.chunks, top_k=top_k)
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
            },
        )


def _merge_results(
    raw_results: list[tuple[str, RetrievalResult, dict[str, object]]],
    *,
    chunks: list[LocalChunkRecord],
    top_k: int,
) -> list[QueryRetrievalResult]:
    best_by_chunk: dict[str, tuple[str, RetrievalResult, dict[str, object]]] = {}
    for subquery_id, result, trace in raw_results:
        current = best_by_chunk.get(result.chunk_id)
        if current is None or _result_score(result) > _result_score(current[1]):
            best_by_chunk[result.chunk_id] = (subquery_id, result, trace)

    ordered = sorted(best_by_chunk.values(), key=lambda item: _result_score(item[1]), reverse=True)
    diversified: list[tuple[str, RetrievalResult, dict[str, object]]] = []
    seen_subqueries: set[str] = set()
    for item in ordered:
        subquery_id = item[0]
        if subquery_id in seen_subqueries:
            continue
        diversified.append(item)
        seen_subqueries.add(subquery_id)
    diversified_ids = {item[1].chunk_id for item in diversified}
    diversified.extend(item for item in ordered if item[1].chunk_id not in diversified_ids)
    merged: list[QueryRetrievalResult] = []
    for rank, (subquery_id, result, trace) in enumerate(diversified[:top_k], start=1):
        label = f"S{rank}"
        parent = hydrate_parent_context(result, chunks)
        metadata = dict(result.metadata)
        metadata["citation_label"] = label
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
                trace=trace,
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
