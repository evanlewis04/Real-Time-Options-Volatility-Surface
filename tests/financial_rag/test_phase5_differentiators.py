import json
from pathlib import Path

from src.financial_rag.api import LocalRagApiService, QueryRequest
from src.financial_rag.differentiators import (
    check_local_companyfacts,
    detect_filing_changes,
    get_market_context,
    score_language_signals,
    summarize_language_signals,
)
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever
from src.financial_rag.workbench import change_rows, language_signal_rows


def test_filing_change_detection_outputs_stable_records() -> None:
    chunks = [
        _chunk("old", "Revenue may decline.\n\nSupply constraints may affect demand.", date="2025-02-01"),
        _chunk("new", "Revenue may grow.\n\nSupply constraints and regulation may affect demand.", date="2026-02-01"),
    ]

    first = detect_filing_changes(chunks, ticker="NVDA")
    second = detect_filing_changes(chunks, ticker="NVDA")

    assert first
    assert [record.change_id for record in first] == [record.change_id for record in second]
    assert first[0].ticker == "NVDA"
    assert first[0].item_number == "1A"
    assert first[0].previous_accession == "old-accession"
    assert first[0].current_accession == "new-accession"
    assert first[0].previous_source_url.startswith("https://www.sec.gov")


def test_xbrl_unavailable_and_local_fixture_behavior(tmp_path: Path) -> None:
    missing = check_local_companyfacts(ticker="NVDA", fact_name="Revenues", facts_path=tmp_path / "missing.json")
    fixture = tmp_path / "NVDA.json"
    fixture.write_text(
        json.dumps(
            {
                "facts": {
                    "us-gaap": {
                        "Revenues": {
                            "units": {
                                "USD": [
                                    {"end": "2025-01-31", "val": 100},
                                    {"end": "2026-01-31", "val": 125},
                                ]
                            }
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = check_local_companyfacts(ticker="NVDA", fact_name="Revenues", facts_path=fixture)

    assert missing.status == "unavailable"
    assert loaded.status == "ok"
    assert loaded.value == 125
    assert loaded.unit == "USD"
    assert loaded.end_date == "2026-01-31"


def test_language_signal_scoring_is_transparent_and_groupable() -> None:
    chunks = [
        _chunk("a", "Risk may increase but demand is strong.", document_id="doc-a"),
        _chunk("b", "Adverse regulation could cause decline.", document_id="doc-a"),
    ]

    chunk_scores = score_language_signals(chunks)
    grouped = summarize_language_signals(chunks, group_by="document_id")

    assert chunk_scores[0].uncertainty_hits == 1
    assert chunk_scores[0].risk_hits == 1
    assert chunk_scores[0].positive_hits == 3
    assert grouped[0].chunk_count == 2
    assert grouped[0].negative_hits >= 1


def test_market_context_fallback_and_provider_payload() -> None:
    fallback = get_market_context("NVDA")
    ok = get_market_context(
        "NVDA",
        provider=lambda ticker: {
            "source_mode": "Fallback",
            "message": f"{ticker} market context",
            "metrics": {"iv_rank": 42},
        },
    )

    assert fallback.status == "unavailable"
    assert ok.status == "ok"
    assert ok.metrics == {"iv_rank": 42}


def test_api_and_workbench_expose_phase5_payloads(tmp_path: Path) -> None:
    facts_dir = tmp_path / "data" / "companyfacts"
    facts_dir.mkdir(parents=True)
    (facts_dir / "NVDA.json").write_text(
        json.dumps({"facts": {"us-gaap": {"Revenues": {"units": {"USD": [{"end": "2026", "val": 1}]}}}}}),
        encoding="utf-8",
    )
    service = _service(root=tmp_path)

    query = service.query(QueryRequest(question="What does NVDA Item 1A say about risk?", top_k=2))
    differentiators = service.differentiators(ticker="NVDA")

    assert query["language_signals"]
    assert query["market_context"]["status"] == "unavailable"
    assert differentiators["xbrl"]["status"] == "ok"
    assert change_rows(differentiators)
    assert language_signal_rows(differentiators)


class _FakeEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _service(*, root: Path) -> LocalRagApiService:
    chunks = [
        _chunk("old", "Risk may decline.", date="2025-02-01"),
        _chunk("new", "Risk may improve.", date="2026-02-01"),
    ]
    retriever = LocalDenseRetriever(
        chunks=chunks,
        embeddings={"old": [0.9, 0.1], "new": [1.0, 0.0]},
        query_embedder=_FakeEmbedder(),
    )
    return LocalRagApiService(chunks=chunks, retriever=retriever, root=root)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    date: str = "2026-02-01",
    document_id: str | None = None,
) -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id or f"doc-{chunk_id}",
            "ticker": "NVDA",
            "form_type": "10-K",
            "filing_date": date,
            "accession_number": f"{chunk_id}-accession",
            "source_url": f"https://www.sec.gov/Archives/{chunk_id}.htm",
            "document_role": "primary",
            "exhibit_type": "",
            "item_number": "1A",
            "speaker_name": "",
            "speaker_role": "",
            "start_offset": 0,
            "end_offset": len(text),
        },
    )
