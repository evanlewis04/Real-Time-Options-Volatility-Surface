from pathlib import Path

from src.financial_rag.api import LocalRagApiService, QueryRequest, serialize_chunk
from src.financial_rag.evaluation import build_retrieval_eval_report, write_retrieval_eval_report
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever
from src.financial_rag.workbench import coverage_rows, evidence_rows, rejected_citation_rows


def test_api_health_coverage_chunk_and_query_payloads() -> None:
    service = _service()

    health = service.health()
    coverage = service.coverage(tickers=["NVDA"])
    chunk = service.get_chunk("risk")
    query = service.query(QueryRequest(question="What does NVDA Item 1A say about risk?", top_k=2))

    assert health["status"] == "ok"
    assert health["chunk_count"] == 2
    assert coverage["tickers"]["NVDA"]["form_types"] == ["10-K", "EX-99"]
    assert chunk is not None
    assert chunk["chunk_id"] == "risk"
    assert query["routed_query"]["query_type"] == "single_doc_lookup"
    assert query["results"][0]["citation_label"] == "S1"
    assert query["results"][0]["metadata"]["ticker"] == "NVDA"
    assert query["results"][0]["parent_context"]["context_chunk_ids"]
    assert query["citations"]["accepted"][0]["chunk_id"] == "risk"
    assert query["citations"]["rejected"] == []


def test_workbench_rows_flatten_evidence_coverage_and_rejected_citations() -> None:
    service = _service()
    payload = service.query(QueryRequest(question="What does NVDA Item 1A say about risk?", top_k=1))
    payload["citations"]["rejected"] = ["S9"]

    evidence = evidence_rows(payload)
    coverage = coverage_rows(payload["coverage"])
    rejected = rejected_citation_rows(payload)

    assert evidence[0]["label"] == "S1"
    assert evidence[0]["ticker"] == "NVDA"
    assert evidence[0]["chunk_id"] == "risk"
    assert coverage[0]["ticker"] == "NVDA"
    assert "10-K" in coverage[0]["forms"]
    assert rejected == [{"label": "S9"}]


def test_eval_report_marks_unlabeled_and_writes_ignored_artifact(tmp_path: Path) -> None:
    from src.financial_rag.evaluation import RetrievalEvalCase

    report = build_retrieval_eval_report(
        [
            RetrievalEvalCase("labeled", "Risk?", relevant_chunk_ids={"risk"}),
            RetrievalEvalCase("unlabeled", "Temporal risk?"),
        ],
        {"labeled": ["risk"]},
        k=5,
    )
    output = write_retrieval_eval_report(report, tmp_path / "artifacts" / "rag_eval" / "report.json")

    assert report["labeled_case_count"] == 1
    assert report["unlabeled_case_count"] == 1
    assert report["mean_recall_at_k"] == 1.0
    assert report["mrr"] == 1.0
    assert report["unlabeled_queries"][0]["status"] == "unlabeled"
    assert output.exists()


def test_serialize_chunk_payload() -> None:
    chunk = _chunk("risk", "risk text")

    payload = serialize_chunk(chunk)

    assert payload["chunk_id"] == "risk"
    assert payload["chunk_text"] == "risk text"
    assert payload["metadata"]["source_url"].startswith("https://www.sec.gov")


class _FakeEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _service() -> LocalRagApiService:
    chunks = [
        _chunk("risk", "risk disclosure", form_type="10-K", item_number="1A"),
        _chunk(
            "cfo",
            "cfo commentary",
            form_type="EX-99",
            document_role="exhibit",
            exhibit_type="CFO_COMMENTARY",
        ),
    ]
    retriever = LocalDenseRetriever(
        chunks=chunks,
        embeddings={"risk": [1.0, 0.0], "cfo": [0.2, 0.8]},
        query_embedder=_FakeEmbedder(),
    )
    return LocalRagApiService(chunks=chunks, retriever=retriever)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    form_type: str = "10-K",
    document_role: str = "primary",
    exhibit_type: str = "",
    item_number: str = "",
) -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "doc",
            "ticker": "NVDA",
            "form_type": form_type,
            "filing_date": "2026-02-25",
            "accession_number": "0001045810-26-000021",
            "source_url": "https://www.sec.gov/Archives/doc.htm",
            "document_role": document_role,
            "exhibit_type": exhibit_type,
            "item_number": item_number,
            "speaker_name": "",
            "speaker_role": "",
            "start_offset": 0 if chunk_id == "risk" else 100,
            "end_offset": 50 if chunk_id == "risk" else 150,
        },
    )
