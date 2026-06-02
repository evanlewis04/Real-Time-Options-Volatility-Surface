"""Tests for the opt-in rerank stage.

The default platform path uses no reranker (it does not beat the domain-tuned
first stage on the current eval). These tests guard the rerank infrastructure:
the lexical reranker ranks relevant text above weak text and is deterministic,
and the protected-set wiring never drops a first-stage result from the final
top-k (so source-hit and recall cannot regress when a reranker is enabled).
"""

import pytest

from src.financial_rag.query.pipeline import QueryPipeline
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever
from src.financial_rag.retrieval.rerank import LexicalRerankerV1, RerankCandidate, build_reranker


def test_lexical_reranker_scores_relevant_above_weak() -> None:
    reranker = LexicalRerankerV1(
        ["unrelated cooking recipe text", "alpha beta gamma", "data center demand and supply commentary"]
    )
    candidates = [
        RerankCandidate("relevant", "Data center demand accelerated across the quarter.", {}),
        RerankCandidate("weak", "An unrelated cooking recipe with onions.", {}),
    ]

    scores = reranker.score("What does the company say about data center demand?", candidates)

    assert scores[0] > scores[1]


def test_lexical_reranker_is_deterministic() -> None:
    reranker = LexicalRerankerV1(["data center demand", "supply constraints", "gross margin"])
    candidates = [
        RerankCandidate("a", "Data center demand grew.", {}),
        RerankCandidate("b", "Supply constraints persisted.", {}),
    ]

    assert reranker.score("data center demand", candidates) == reranker.score("data center demand", candidates)


def test_build_reranker_dispatch() -> None:
    assert build_reranker("none", chunk_texts=[]) is None
    assert isinstance(build_reranker("lexical", chunk_texts=["a b c"]), LexicalRerankerV1)
    with pytest.raises(ValueError):
        build_reranker("bogus", chunk_texts=["a b c"])


def test_protected_set_rerank_preserves_first_stage_result_set() -> None:
    chunks = [
        _chunk("c1", "Data center demand accelerated and revenue grew."),
        _chunk("c2", "Supply constraints and export controls affected results."),
        _chunk("c3", "Gross margin improved on product mix."),
        _chunk("c4", "Inventory and capacity commitments increased."),
    ]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}
    retriever = LocalDenseRetriever(chunks=chunks, embeddings=embeddings, query_embedder=_FakeEmbedder())

    baseline = QueryPipeline(retriever=retriever, chunks=chunks)
    reranked = QueryPipeline(
        retriever=retriever,
        chunks=chunks,
        reranker=LexicalRerankerV1([chunk.chunk_text for chunk in chunks]),
    )

    base_ids = {result.chunk_id for result in baseline.run("data center demand", top_k=3).results}
    rerank_ids = {result.chunk_id for result in reranked.run("data center demand", top_k=3).results}

    # The reranker only reorders the protected first-stage set; it never changes
    # which chunks are surfaced, so recall and source-hit cannot regress.
    assert base_ids == rerank_ids


class _FakeEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _chunk(chunk_id: str, text: str) -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "doc",
            "ticker": "NVDA",
            "form_type": "10-K",
            "filing_date": "2026-02-25",
            "accession_number": "0001045810-26-000021",
            "source_url": f"https://www.sec.gov/Archives/{chunk_id}.htm",
            "document_role": "primary",
            "exhibit_type": "",
            "item_number": "1A",
        },
    )
