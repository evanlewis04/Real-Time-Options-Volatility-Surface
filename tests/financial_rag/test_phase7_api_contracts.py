import importlib
import json
from pathlib import Path

import pytest

from src.financial_rag.api import (
    LocalApiError,
    LocalRagApiService,
    QueryRequest,
    api_endpoint_manifest,
    call_local_api_endpoint,
    create_fastapi_app,
    run_api_smoke,
)
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever


def test_endpoint_manifest_covers_phase7_contracts() -> None:
    endpoints = {(endpoint["method"], endpoint["path"]) for endpoint in api_endpoint_manifest()}

    assert ("GET", "/health") in endpoints
    assert ("GET", "/companies") in endpoints
    assert ("GET", "/coverage") in endpoints
    assert ("GET", "/documents") in endpoints
    assert ("POST", "/query") in endpoints
    assert ("GET", "/chunks/{chunk_id}") in endpoints
    assert ("GET", "/differentiators/{ticker}") in endpoints
    assert ("GET", "/market-context/{ticker}") in endpoints


def test_companies_documents_and_market_context_success_payloads() -> None:
    service = _service()

    companies = service.companies()
    documents = service.documents()
    nvda_documents = service.documents(ticker="NVDA")
    market_context = service.market_context(ticker="NVDA")

    assert companies["company_count"] == len(companies["companies"]) == 1
    nvda = companies["companies"][0]
    assert nvda["ticker"] == "NVDA"
    assert nvda["chunk_count"] == 2
    assert nvda["document_count"] == 1
    assert {"form_types", "ex99_chunk_count", "has_press_release", "gaps"} <= set(nvda)

    assert documents["document_count"] == 1
    assert documents["documents"][0]["chunk_count"] == 2
    assert documents["documents"][0]["ticker"] == "NVDA"
    assert nvda_documents["documents"] == documents["documents"]

    assert market_context["ticker"] == "NVDA"
    assert "market_context" in market_context


def test_new_read_endpoints_via_framework_neutral_dispatch() -> None:
    service = _service()

    companies = call_local_api_endpoint(service, method="GET", path="/companies")
    documents = call_local_api_endpoint(service, method="GET", path="/documents", query_params={"ticker": "NVDA"})
    market_context = call_local_api_endpoint(service, method="GET", path="/market-context/NVDA")
    bad_ticker = call_local_api_endpoint(service, method="GET", path="/market-context/ZZZZ")
    bad_documents = call_local_api_endpoint(service, method="GET", path="/documents", query_params={"ticker": "ZZZZ"})

    assert companies["payload"]["company_count"] == 1
    assert documents["payload"]["documents"][0]["ticker"] == "NVDA"
    assert market_context["payload"]["ticker"] == "NVDA"
    assert bad_ticker["status_code"] == 400
    assert bad_ticker["payload"]["error"]["code"] == "unsupported_ticker"
    assert bad_documents["status_code"] == 400
    assert bad_documents["payload"]["error"]["code"] == "unsupported_ticker"


def test_local_api_contract_success_payloads() -> None:
    service = _service()

    health = service.health()
    coverage = service.coverage(tickers=["NVDA"])
    query = service.query(QueryRequest(question="What does NVDA Item 1A say about risk?", ticker="NVDA", top_k=2))
    chunk = service.require_chunk("risk")
    differentiators = service.differentiators(ticker="NVDA")

    assert health["status"] == "ok"
    assert health["service"] == "financial_rag_local_api"
    assert coverage["tickers"]["NVDA"]["chunk_count"] == 2
    assert query["results"][0]["citation_label"] == "S1"
    assert query["citations"]["accepted"][0]["source_url"].startswith("https://www.sec.gov")
    assert chunk["chunk_id"] == "risk"
    assert differentiators["ticker"] == "NVDA"
    assert {"changes", "language_signals", "xbrl", "market_context"} <= set(differentiators)


def test_framework_neutral_endpoint_contracts_use_literal_paths() -> None:
    service = _service()

    health = call_local_api_endpoint(service, method="GET", path="/health")
    coverage = call_local_api_endpoint(service, method="GET", path="/coverage", query_params={"ticker": "NVDA"})
    query = call_local_api_endpoint(
        service,
        method="POST",
        path="/query",
        body={"question": "What risks does NVIDIA describe?", "ticker": "NVDA", "top_k": 2},
    )
    chunk = call_local_api_endpoint(service, method="GET", path="/chunks/risk")
    differentiators = call_local_api_endpoint(service, method="GET", path="/differentiators/NVDA")
    missing = call_local_api_endpoint(service, method="GET", path="/missing")

    assert health["status_code"] == 200
    assert coverage["payload"]["tickers"]["NVDA"]["chunk_count"] == 2
    assert query["payload"]["results"]
    assert chunk["payload"]["chunk_id"] == "risk"
    assert differentiators["payload"]["ticker"] == "NVDA"
    assert missing["status_code"] == 404
    assert missing["payload"]["error"]["code"] == "endpoint_not_found"


@pytest.mark.parametrize(
    ("query_request", "code"),
    [
        (QueryRequest(question="", ticker="NVDA"), "invalid_question"),
        (QueryRequest(question="Risk?", ticker="ZZZZ"), "unsupported_ticker"),
        (QueryRequest(question="Risk?", ticker="NVDA", top_k=0), "invalid_top_k"),
        (QueryRequest(question="Risk?", ticker="NVDA", per_subquery_k=99), "invalid_per_subquery_k"),
    ],
)
def test_query_contract_errors_are_structured(query_request: QueryRequest, code: str) -> None:
    service = _service()

    with pytest.raises(LocalApiError) as exc_info:
        service.query(query_request)

    assert exc_info.value.code == code
    assert exc_info.value.to_dict()["error"]["code"] == code


def test_service_errors_for_missing_cache_invalid_chunk_and_empty_results() -> None:
    empty = _service(chunks=[], embeddings={})
    lexical_fallback = _service(embeddings={})

    assert empty.health()["status"] == "empty_cache"
    with pytest.raises(LocalApiError, match="No local chunks"):
        empty.query(QueryRequest(question="Risk?", ticker="NVDA"))
    with pytest.raises(LocalApiError) as missing_chunk:
        lexical_fallback.require_chunk("missing")
    with pytest.raises(LocalApiError) as empty_results:
        lexical_fallback.query(QueryRequest(question="Risk?", ticker="AMD"))

    assert lexical_fallback.query(QueryRequest(question="Risk?", ticker="NVDA"))["results"]
    assert missing_chunk.value.code == "chunk_not_found"
    assert empty_results.value.code == "empty_retrieval_results"


def test_api_smoke_runs_and_writes_json_artifact(tmp_path: Path) -> None:
    report = run_api_smoke(
        root=tmp_path,
        service=_service(root=tmp_path),
        output_path=tmp_path / "artifacts" / "rag_eval" / "phase7_api_smoke.json",
    )
    loaded = json.loads(Path(report.artifact_path).read_text(encoding="utf-8"))

    assert report.status in {"pass", "warning"}
    assert {step.name for step in report.steps} >= {
        "health",
        "companies",
        "coverage",
        "documents",
        "query",
        "chunk_lookup",
        "differentiators",
        "market_context",
        "readiness",
        "evidence_quality",
    }
    assert loaded["endpoint_count"] == len(api_endpoint_manifest())


def test_fastapi_missing_dependency_has_clear_guidance() -> None:
    try:
        import fastapi  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="FastAPI is not installed"):
            create_fastapi_app(_service())


def test_phase7_keeps_phase1_to_phase6_command_modules_importable() -> None:
    modules = [
        "scripts.financial_rag_phase1_smoke",
        "scripts.financial_rag_phase2_retrieval_smoke",
        "scripts.financial_rag_phase3_query_smoke",
        "scripts.financial_rag_phase4_workbench_smoke",
        "scripts.financial_rag_phase5_differentiators_report",
        "scripts.financial_rag_phase6_demo_workflow",
        "scripts.financial_rag_phase7_api_smoke",
        "scripts.financial_rag_phase7_api_server",
    ]

    for module in modules:
        assert importlib.import_module(module)


class _FakeEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _service(
    *,
    root: Path = Path("."),
    chunks: list[LocalChunkRecord] | None = None,
    embeddings: dict[str, list[float]] | None = None,
) -> LocalRagApiService:
    local_chunks = chunks if chunks is not None else [
        _chunk("risk", "Risk factors include demand and export controls.", item_number="1A"),
        _chunk("release", "NVIDIA announced strong demand.", form_type="EX-99", document_role="exhibit"),
    ]
    local_embeddings = embeddings if embeddings is not None else {"risk": [1.0, 0.0], "release": [0.6, 0.4]}
    retriever = LocalDenseRetriever(chunks=local_chunks, embeddings=local_embeddings, query_embedder=_FakeEmbedder())
    return LocalRagApiService(chunks=local_chunks, retriever=retriever, root=root)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    form_type: str = "10-K",
    document_role: str = "primary",
    item_number: str = "1A",
) -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "doc",
            "ticker": "NVDA",
            "form_type": form_type,
            "filing_date": "2026-02-01",
            "accession_number": f"{chunk_id}-accession",
            "source_url": f"https://www.sec.gov/Archives/{chunk_id}.htm",
            "document_role": document_role,
            "exhibit_type": "PRESS_RELEASE" if document_role == "exhibit" else "",
            "item_number": item_number,
            "start_offset": 0 if chunk_id == "risk" else 100,
            "end_offset": 50 if chunk_id == "risk" else 150,
        },
    )
