"""Phase 1 Stage 2: point-in-time as-of query filter and its look-ahead invariant.

The headline guarantee: an as-of query never returns a record filed after the
as-of instant. That is the whole look-ahead-bias defense in one assertion, so it
is exercised here as a property over random as-of dates on a synthetic corpus
with known filed_at timestamps — the deterministic vehicle Planner specified,
since the real corpus dates cannot be controlled.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from src.financial_rag.api.local_service import LocalApiError, QueryRequest, _parse_as_of
from src.financial_rag.query import QueryPipeline
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever


class _ConstantEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _chunk(chunk_id: str, filed_at: str, *, ticker: str = "NVDA") -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text="revenue growth commentary and outlook",
        metadata={
            "chunk_id": chunk_id,
            "document_id": f"doc-{chunk_id}",
            "ticker": ticker,
            "form_type": "10-K",
            "document_role": "primary",
            "filing_date": (filed_at[:10] or "2026-01-01"),
            "filed_at": filed_at,
            "period_end": "2025-12-31",
            "accession_number": f"0001045810-26-0000{chunk_id[-2:]}",
            "source_url": f"https://www.sec.gov/Archives/{chunk_id}.htm",
        },
    )


def _dated_corpus(count: int = 24) -> tuple[list[LocalChunkRecord], dict[str, list[float]]]:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    chunks: list[LocalChunkRecord] = []
    embeddings: dict[str, list[float]] = {}
    for i in range(count):
        chunk_id = f"c{i:02d}"
        filed_at = (base + timedelta(days=i * 5, hours=i)).isoformat()
        chunks.append(_chunk(chunk_id, filed_at))
        embeddings[chunk_id] = [1.0, 0.0]
    return chunks, embeddings


def _retriever(chunks: list[LocalChunkRecord], embeddings: dict[str, list[float]]) -> LocalDenseRetriever:
    return LocalDenseRetriever(chunks=chunks, embeddings=embeddings, query_embedder=_ConstantEmbedder())


def test_as_of_never_returns_a_record_filed_after_the_as_of_instant() -> None:
    # The headline look-ahead invariant, over random as-of dates.
    chunks, embeddings = _dated_corpus()
    retriever = _retriever(chunks, embeddings)
    filed_ats = [datetime.fromisoformat(chunk.metadata["filed_at"]) for chunk in chunks]
    lo, hi = min(filed_ats), max(filed_ats)
    span = (hi - lo).total_seconds()

    rng = random.Random(20260801)
    for _ in range(200):
        as_of = lo + timedelta(seconds=rng.uniform(-span * 0.2, span * 1.2))
        results = retriever.search(query="revenue growth", top_k=len(chunks), as_of=as_of)
        for result in results:
            filed_at = datetime.fromisoformat(result.metadata["filed_at"])
            assert filed_at <= as_of, f"leak: {result.chunk_id} filed {filed_at} > as_of {as_of}"


def test_as_of_boundary_includes_a_record_filed_at_exactly_as_of() -> None:
    chunks, embeddings = _dated_corpus(count=3)
    retriever = _retriever(chunks, embeddings)
    exact = datetime.fromisoformat(chunks[1].metadata["filed_at"])

    returned = {r.chunk_id for r in retriever.search(query="revenue", top_k=10, as_of=exact)}

    assert chunks[1].chunk_id in returned  # filed_at == as_of is knowable (<=)
    assert chunks[2].chunk_id not in returned  # filed later, excluded


def test_as_of_excludes_chunks_with_empty_filed_at() -> None:
    # Every pre-backfill chunk has an empty filed_at; an unknown public date
    # cannot be proven knowable-as-of-D, so it is excluded (data-honesty policy).
    chunks = [
        _chunk("dated", datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()),
        _chunk("undated", ""),
    ]
    embeddings = {"dated": [1.0, 0.0], "undated": [1.0, 0.0]}
    retriever = _retriever(chunks, embeddings)

    as_of = datetime(2026, 6, 1, tzinfo=timezone.utc)
    returned = {r.chunk_id for r in retriever.search(query="revenue", top_k=10, as_of=as_of)}

    assert returned == {"dated"}


def test_as_of_none_is_a_no_op_and_keeps_undated_and_future_chunks() -> None:
    # as_of=None must not filter at all: the eval path relies on this being an
    # exact no-op, so undated and future-dated chunks all remain retrievable.
    future = (datetime.now(timezone.utc) + timedelta(days=3650)).isoformat()
    chunks = [
        _chunk("undated", ""),
        _chunk("future", future),
    ]
    embeddings = {"undated": [1.0, 0.0], "future": [1.0, 0.0]}
    retriever = _retriever(chunks, embeddings)

    returned = {r.chunk_id for r in retriever.search(query="revenue", top_k=10, as_of=None)}

    assert returned == {"undated", "future"}


def test_naive_as_of_is_treated_as_utc() -> None:
    chunks = [
        _chunk("before", datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc).isoformat()),
        _chunk("after", datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc).isoformat()),
    ]
    embeddings = {"before": [1.0, 0.0], "after": [1.0, 0.0]}
    retriever = _retriever(chunks, embeddings)

    naive_noon = datetime(2026, 3, 1, 12, 0)  # no tzinfo -> assumed UTC
    returned = {r.chunk_id for r in retriever.search(query="revenue", top_k=10, as_of=naive_noon)}

    assert returned == {"before"}


def test_as_of_threads_through_the_pipeline_run() -> None:
    chunks, embeddings = _dated_corpus(count=6)
    retriever = _retriever(chunks, embeddings)
    pipeline = QueryPipeline(retriever=retriever, chunks=chunks)
    as_of = datetime.fromisoformat(chunks[2].metadata["filed_at"])

    result = pipeline.run("What did the company report about revenue growth?", default_ticker="NVDA", as_of=as_of)

    assert result.trace["as_of"] == as_of.isoformat()
    for item in result.results:
        filed_at = datetime.fromisoformat(item.metadata["filed_at"])
        assert filed_at <= as_of


def test_query_request_parses_as_of_from_payload_forms() -> None:
    assert _parse_as_of(None) is None
    assert _parse_as_of("") is None
    assert _parse_as_of("2026-03-01") == datetime(2026, 3, 1)
    assert _parse_as_of("2026-03-01T15:00:00Z") == datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc)


def test_malformed_as_of_is_a_client_error_not_a_silent_full_corpus_query() -> None:
    try:
        _parse_as_of("not-a-date")
    except LocalApiError as exc:
        assert exc.status_code == 400
        assert exc.code == "invalid_as_of"
    else:  # pragma: no cover - defensive
        raise AssertionError("expected LocalApiError for malformed as_of")


def test_query_request_defaults_as_of_to_none() -> None:
    assert QueryRequest().as_of is None
