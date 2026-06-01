"""Tests for the thin RAG-evidence + market-context integration prototype."""

from src.financial_rag.api import LocalRagApiService
from src.financial_rag.differentiators import get_market_context
from src.financial_rag.integration import (
    FILING_EVIDENCE_LABEL,
    MARKET_CONTEXT_LABEL,
    build_brief_from_service,
    build_market_evidence_brief,
    market_provider_from_metrics,
)
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever


def test_brief_separates_cited_disclosure_from_market_context_when_available() -> None:
    provider = market_provider_from_metrics(
        {"source_mode": "Fallback", "front_expected_move_pct": 8.2, "iv_rank": 64.0}
    )
    market_context = get_market_context("NVDA", provider=provider)

    brief = build_market_evidence_brief(_query_payload(), market_context, question="Demand?", ticker="nvda")
    payload = brief.to_dict()

    assert payload["ticker"] == "NVDA"
    # Filing evidence is cited disclosure.
    assert payload["filing_evidence"]["source"] == FILING_EVIDENCE_LABEL
    assert payload["filing_evidence"]["result_count"] == 1
    assert payload["filing_evidence"]["accepted_citations"][0]["chunk_id"] == "risk"
    assert payload["filing_evidence"]["evidence"][0]["source_url"].startswith("https://www.sec.gov")
    # Market context is a separate, provenance-labeled block.
    assert payload["market_context"]["source"] == MARKET_CONTEXT_LABEL
    assert payload["market_context"]["status"] == "ok"
    assert payload["market_context"]["source_mode"] == "Fallback"
    assert payload["market_context"]["metrics"]["front_expected_move_pct"] == 8.2
    # Data-source labels stay explicit and distinct.
    labels = {source["label"] for source in payload["data_sources"]}
    assert labels == {FILING_EVIDENCE_LABEL, MARKET_CONTEXT_LABEL}
    assert any("is not filing evidence" in note for note in payload["notes"])


def test_brief_labels_market_context_unavailable_without_provider() -> None:
    market_context = get_market_context("NVDA")  # no provider -> unavailable

    brief = build_market_evidence_brief(_query_payload(), market_context, question="Demand?", ticker="NVDA")
    payload = brief.to_dict()

    assert payload["market_context"]["status"] == "unavailable"
    market_source = next(s for s in payload["data_sources"] if s["label"] == MARKET_CONTEXT_LABEL)
    assert market_source["provenance"] == "unavailable"
    assert any("do not infer market reaction" in note for note in payload["notes"])


def test_build_brief_from_service_runs_retrieval_and_attaches_market_context() -> None:
    service = _service()
    provider = market_provider_from_metrics({"source_mode": "Delayed", "iv_rank": 50.0})

    brief = build_brief_from_service(
        service,
        question="What does NVDA Item 1A say about risk?",
        ticker="NVDA",
        top_k=2,
        market_provider=provider,
    )
    payload = brief.to_dict()

    assert payload["filing_evidence"]["result_count"] >= 1
    assert payload["market_context"]["status"] == "ok"
    assert payload["market_context"]["metrics"]["iv_rank"] == 50.0


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
