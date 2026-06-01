"""Local API smoke workflow for Phase 7 contract hardening."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.financial_rag.api.local_service import (
    LocalApiError,
    LocalRagApiService,
    QueryRequest,
    api_endpoint_manifest,
    build_local_api_service,
)
from src.financial_rag.audit import build_evidence_quality_report, build_readiness_report, write_json_report


DEFAULT_PHASE7_QUERY = "What risks does NVIDIA describe?"


@dataclass(frozen=True)
class ApiSmokeStep:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ApiSmokeReport:
    status: str
    endpoint_count: int
    steps: list[ApiSmokeStep]
    artifact_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload


def run_api_smoke(
    *,
    root: Path | str,
    ticker: str = "NVDA",
    query: str = DEFAULT_PHASE7_QUERY,
    output_path: Path | str | None = None,
    service: LocalRagApiService | None = None,
) -> ApiSmokeReport:
    """Exercise the local service contract without starting a server."""

    root_path = Path(root)
    ticker = ticker.upper()
    svc = service or build_local_api_service(root=root_path, use_voyage=False)
    steps: list[ApiSmokeStep] = []

    health = _run_step("health", lambda: svc.health(), lambda payload: payload.get("status") in {"ok", "empty_cache"})
    steps.append(health)

    coverage = _run_step(
        "coverage",
        lambda: svc.coverage(tickers=[ticker]),
        lambda payload: ticker in payload.get("tickers", {}),
    )
    steps.append(coverage)

    query_payload: dict[str, Any] | None = None

    def _query() -> dict[str, Any]:
        nonlocal query_payload
        query_payload = svc.query(QueryRequest(question=query, ticker=ticker, top_k=5, per_subquery_k=5))
        return query_payload

    query_step = _run_step("query", _query, lambda payload: bool(payload.get("results")))
    steps.append(query_step)

    chunk_id = _first_chunk_id(query_payload) if query_payload else _first_local_chunk_id(svc)
    steps.append(
        _run_step(
            "chunk_lookup",
            lambda: svc.require_chunk(chunk_id),
            lambda payload: payload.get("chunk_id") == chunk_id,
        )
    )

    steps.append(
        _run_step(
            "differentiators",
            lambda: svc.differentiators(ticker=ticker),
            lambda payload: payload.get("ticker") == ticker,
        )
    )

    steps.append(
        _run_step(
            "companies",
            lambda: svc.companies(),
            lambda payload: isinstance(payload.get("companies"), list)
            and payload.get("company_count", 0) == len(payload.get("companies", [])),
        )
    )

    steps.append(
        _run_step(
            "documents",
            lambda: svc.documents(ticker=ticker),
            lambda payload: isinstance(payload.get("documents"), list)
            and payload.get("document_count", 0) == len(payload.get("documents", [])),
        )
    )

    steps.append(
        _run_step(
            "market_context",
            lambda: svc.market_context(ticker=ticker),
            lambda payload: payload.get("ticker") == ticker and "market_context" in payload,
        )
    )

    readiness = build_readiness_report(svc.chunks, svc.retriever.embeddings, tickers=[ticker], root=root_path)
    steps.append(
        ApiSmokeStep(
            name="readiness",
            status="pass" if readiness.status in {"ready", "warning"} else "fail",
            message=f"{readiness.chunk_count} chunks and {readiness.embedding_count} embeddings inspected.",
            details={"readiness_status": readiness.status, "issue_count": len(readiness.issues)},
        )
    )

    if query_payload is not None:
        evidence_quality = build_evidence_quality_report(query_payload)
        steps.append(
            ApiSmokeStep(
                name="evidence_quality",
                status="pass" if evidence_quality.status == "pass" else "warning"
                if evidence_quality.status == "warning"
                else "fail",
                message=f"{evidence_quality.result_count} retrieved chunks checked.",
                details={
                    "evidence_quality_status": evidence_quality.status,
                    "issue_count": len(evidence_quality.issues),
                },
            )
        )

    report = ApiSmokeReport(
        status="fail" if any(step.status == "fail" for step in steps) else "warning"
        if any(step.status == "warning" for step in steps)
        else "pass",
        endpoint_count=len(api_endpoint_manifest()),
        steps=steps,
        artifact_path=str(output_path or ""),
    )
    if output_path:
        path = write_json_report(report.to_dict(), Path(output_path))
        report = ApiSmokeReport(
            status=report.status,
            endpoint_count=report.endpoint_count,
            steps=report.steps,
            artifact_path=str(path),
        )
    return report


def _run_step(
    name: str,
    operation: Callable[[], dict[str, Any]],
    is_success: Callable[[dict[str, Any]], bool],
) -> ApiSmokeStep:
    try:
        payload = operation()
    except LocalApiError as exc:
        return ApiSmokeStep(
            name=name,
            status="fail",
            message=exc.message,
            details={"code": exc.code, "status_code": exc.status_code, **exc.details},
        )
    status = "pass" if is_success(payload) else "fail"
    return ApiSmokeStep(name=name, status=status, message=f"{name} contract exercised.", details=_summary(payload))


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    if "chunk_count" in payload:
        return {"chunk_count": payload.get("chunk_count", 0), "embedding_count": payload.get("embedding_count", 0)}
    if "tickers" in payload:
        return {"tickers": sorted(payload.get("tickers", {}))}
    if "results" in payload:
        return {"result_count": len(payload.get("results", []))}
    if "changes" in payload:
        return {"changes": len(payload.get("changes", [])), "language_signals": len(payload.get("language_signals", []))}
    if "chunk_id" in payload:
        return {"chunk_id": payload.get("chunk_id", "")}
    return {}


def _first_chunk_id(query_payload: dict[str, Any] | None) -> str:
    if not query_payload:
        return ""
    for result in query_payload.get("results", []):
        chunk_id = str(result.get("chunk_id", ""))
        if chunk_id:
            return chunk_id
    return ""


def _first_local_chunk_id(service: LocalRagApiService) -> str:
    return service.chunks[0].chunk_id if service.chunks else ""
