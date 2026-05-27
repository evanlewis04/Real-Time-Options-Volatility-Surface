import importlib
import json
from pathlib import Path

from src.financial_rag.api import LocalRagApiService, QueryRequest
from src.financial_rag.audit import build_evidence_quality_report, build_readiness_report, write_json_report
from src.financial_rag.demo import run_demo_workflow
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever
from src.financial_rag.workbench import evidence_quality_issue_rows, readiness_issue_rows


def test_readiness_audit_reports_cache_gaps_and_writes_json(tmp_path: Path) -> None:
    chunks = [
        _chunk("risk", "risk text", item_number=""),
        _chunk("release", "press release", form_type="EX-99", document_role="exhibit", exhibit_type="PRESS_RELEASE"),
        _chunk("other", "unsupported", ticker="ZZZZ", item_number=""),
    ]
    report = build_readiness_report(
        chunks,
        {"risk": [1.0, 0.0]},
        tickers=["NVDA", "ZZZZ"],
        root=tmp_path,
        supported_tickers={"NVDA"},
    )
    path = write_json_report(report.to_dict(), tmp_path / "artifacts" / "rag_eval" / "readiness.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert report.status == "fail"
    assert report.missing_embedding_count == 2
    assert report.missing_item_metadata_count == 2
    assert report.unsupported_tickers == ["ZZZZ"]
    assert report.companyfacts_available["NVDA"] is False
    assert loaded["missing_embedding_count"] == 2
    assert readiness_issue_rows(report.to_dict())


def test_evidence_quality_checks_metadata_citations_parent_context_and_duplicates() -> None:
    payload = {
        "results": [
            {
                "chunk_id": "risk",
                "citation_label": "S1",
                "source_url": "",
                "metadata": {"ticker": "NVDA", "filing_date": "2026-02-01", "accession_number": ""},
                "parent_context": None,
            },
            {
                "chunk_id": "risk",
                "citation_label": "S2",
                "source_url": "https://www.sec.gov/Archives/risk.htm",
                "metadata": {"ticker": "", "filing_date": "", "accession_number": "0001"},
                "parent_context": {"context_chunk_ids": ["risk"]},
            },
        ],
        "citations": {"accepted": [{"label": "S9"}], "rejected": ["S8"]},
    }

    report = build_evidence_quality_report(payload)

    assert report.status == "fail"
    assert report.duplicate_chunk_ids == ["risk"]
    assert report.missing_url_count == 1
    assert report.missing_accession_count == 1
    assert report.missing_date_count == 1
    assert report.missing_ticker_count == 1
    assert report.missing_parent_context_count == 1
    assert report.invalid_citation_count == 2
    assert evidence_quality_issue_rows(report.to_dict())


def test_demo_workflow_runs_local_smoke_and_writes_artifacts(tmp_path: Path) -> None:
    service = _service(root=tmp_path)

    report = run_demo_workflow(root=tmp_path, service=service, output_dir=tmp_path / "artifacts" / "rag_eval")

    assert report.status in {"pass", "warning"}
    assert {step.name for step in report.steps} == {
        "local_cache_readiness",
        "query_smoke",
        "workbench_smoke",
        "differentiator_report",
        "eval_report",
    }
    assert Path(report.artifact_paths["readiness"]).exists()
    assert Path(report.artifact_paths["evidence_quality"]).exists()
    assert Path(report.artifact_paths["eval_report"]).exists()


def test_phase6_keeps_phase1_to_phase5_command_modules_importable() -> None:
    modules = [
        "scripts.financial_rag_phase1_smoke",
        "scripts.financial_rag_phase2_retrieval_smoke",
        "scripts.financial_rag_phase3_query_smoke",
        "scripts.financial_rag_phase4_workbench_smoke",
        "scripts.financial_rag_phase5_differentiators_report",
        "scripts.financial_rag_phase6_demo_workflow",
    ]

    for module in modules:
        assert importlib.import_module(module)


def test_phase6_workbench_rows_stay_compatible_with_query_payload() -> None:
    service = _service(root=Path("."))
    payload = service.query(QueryRequest(question="What does NVDA Item 1A say about risk?", top_k=1))
    readiness = build_readiness_report(service.chunks, service.retriever.embeddings, tickers=["NVDA"])
    quality = build_evidence_quality_report(payload)

    assert isinstance(readiness_issue_rows(readiness.to_dict()), list)
    assert isinstance(evidence_quality_issue_rows(quality.to_dict()), list)


class _FakeEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _service(*, root: Path) -> LocalRagApiService:
    chunks = [
        _chunk("risk", "Risk factors include demand and export controls.", item_number="1A"),
        _chunk("release", "NVIDIA announced strong demand.", form_type="EX-99", document_role="exhibit"),
    ]
    retriever = LocalDenseRetriever(
        chunks=chunks,
        embeddings={"risk": [1.0, 0.0], "release": [0.6, 0.4]},
        query_embedder=_FakeEmbedder(),
    )
    return LocalRagApiService(chunks=chunks, retriever=retriever, root=root)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    ticker: str = "NVDA",
    form_type: str = "10-K",
    document_role: str = "primary",
    exhibit_type: str = "",
    item_number: str = "1A",
) -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "doc",
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": "2026-02-01",
            "accession_number": f"{chunk_id}-accession",
            "source_url": f"https://www.sec.gov/Archives/{chunk_id}.htm",
            "document_role": document_role,
            "exhibit_type": exhibit_type,
            "item_number": item_number,
            "start_offset": 0 if chunk_id == "risk" else 100,
            "end_offset": 50 if chunk_id == "risk" else 150,
        },
    )
