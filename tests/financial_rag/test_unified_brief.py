"""Tests for the unified analyst brief (Milestone B)."""

from src.financial_rag.api import LocalRagApiService
from src.financial_rag.differentiators import get_market_context
from src.financial_rag.integration import (
    FILING_EVIDENCE_LABEL,
    MARKET_CONTEXT_LABEL,
    assemble_unified_brief,
    build_unified_brief,
    market_provider_from_metrics,
)
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever


def test_assemble_brief_combines_evidence_market_and_gated_answer() -> None:
    provider = market_provider_from_metrics({"source_mode": "Fallback", "front_expected_move_pct": 8.2})
    market_context = get_market_context("NVDA", provider=provider)
    answer = {
        "answer_text": "Data center demand accelerated [S1].",
        "accepted_citations": [{"label": "S1", "chunk_id": "risk"}],
        "rejected_citations": [],
        "dry_run": False,
    }

    brief = assemble_unified_brief(
        _query_payload(),
        market_context,
        question="Demand?",
        ticker="nvda",
        evidence_quality_status="pass",
        openai_ready=True,
        answer=answer,
    ).to_dict()

    assert brief["ticker"] == "NVDA"
    assert brief["answer"]["answer_text"].endswith("[S1].")
    assert brief["answer_gate"]["allowed"] is True
    assert brief["filing_evidence"]["source"] == FILING_EVIDENCE_LABEL
    assert brief["filing_evidence"]["accepted_citations"][0]["chunk_id"] == "risk"
    assert brief["market_context"]["source"] == MARKET_CONTEXT_LABEL
    assert brief["market_context"]["source_mode"] == "Fallback"
    assert {source["label"] for source in brief["data_sources"]} == {FILING_EVIDENCE_LABEL, MARKET_CONTEXT_LABEL}
    assert any("not market advice" in note for note in brief["notes"])


def test_assemble_brief_blocks_answer_when_openai_not_ready() -> None:
    market_context = get_market_context("NVDA")  # unavailable

    brief = assemble_unified_brief(
        _query_payload(),
        market_context,
        question="Demand?",
        ticker="NVDA",
        evidence_quality_status="pass",
        openai_ready=False,
        openai_issues=["OPENAI_API_KEY is missing."],
    ).to_dict()

    assert brief["answer"] is None
    assert brief["answer_gate"]["allowed"] is False
    assert "OPENAI_API_KEY is missing." in brief["answer_gate"]["reasons"]
    assert brief["market_context"]["status"] == "unavailable"
    assert any("primary output" in note for note in brief["notes"])


def test_build_unified_brief_runs_retrieval_without_calling_openai() -> None:
    service = _service()
    provider = market_provider_from_metrics({"source_mode": "Delayed", "iv_rank": 50.0})

    brief = build_unified_brief(
        service,
        question="What does NVDA Item 1A say about risk?",
        ticker="NVDA",
        top_k=2,
        market_provider=provider,
        run_answer=False,
    ).to_dict()

    assert brief["filing_evidence"]["result_count"] >= 1
    assert brief["market_context"]["metrics"]["iv_rank"] == 50.0
    # run_answer is False, so no answer is generated regardless of readiness.
    assert brief["answer"] is None


def _query_payload() -> dict[str, object]:
    return {
        "results": [
            {
                "chunk_id": "risk",
                "citation_label": "S1",
                "source_url": "https://www.sec.gov/Archives/risk.htm",
                "source_excerpt": "Data center demand accelerated.",
                "metadata": {"ticker": "NVDA", "form_type": "10-K", "filing_date": "2026-02-25"},
            }
        ],
        "citations": {"accepted": [{"label": "S1", "chunk_id": "risk"}], "rejected": []},
    }


class _FakeEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _service() -> LocalRagApiService:
    chunks = [
        LocalChunkRecord(
            chunk_id="risk",
            chunk_text="Risk factors include data center demand and export controls.",
            metadata={
                "chunk_id": "risk",
                "document_id": "doc",
                "ticker": "NVDA",
                "form_type": "10-K",
                "filing_date": "2026-02-25",
                "accession_number": "0001045810-26-000021",
                "source_url": "https://www.sec.gov/Archives/doc.htm",
                "document_role": "primary",
                "exhibit_type": "",
                "item_number": "1A",
            },
        )
    ]
    retriever = LocalDenseRetriever(chunks=chunks, embeddings={"risk": [1.0, 0.0]}, query_embedder=_FakeEmbedder())
    return LocalRagApiService(chunks=chunks, retriever=retriever)
