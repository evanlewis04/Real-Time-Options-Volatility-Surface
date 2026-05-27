"""End-to-end local demo workflow orchestration for Phase 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.financial_rag.api import LocalApiError, LocalRagApiService, QueryRequest, build_local_api_service
from src.financial_rag.audit import build_evidence_quality_report, build_readiness_report, write_json_report
from src.financial_rag.evaluation import PHASE3_ROUTED_RETRIEVAL_FIXTURES, build_retrieval_eval_report
from src.financial_rag.workbench import change_rows, coverage_rows, evidence_rows, language_signal_rows


DEFAULT_PHASE6_QUERY = "What risks does NVIDIA describe?"


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DemoWorkflowReport:
    status: str
    prerequisites: list[str]
    steps: list[WorkflowStep]
    artifact_paths: dict[str, str]
    next_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload


def run_demo_workflow(
    *,
    root: Path | str,
    ticker: str = "NVDA",
    query: str = DEFAULT_PHASE6_QUERY,
    output_dir: Path | str | None = None,
    service: LocalRagApiService | None = None,
) -> DemoWorkflowReport:
    """Run a local-only demo workflow and write ignored JSON artifacts."""

    root_path = Path(root)
    artifacts_dir = Path(output_dir) if output_dir else root_path / "artifacts" / "rag_eval"
    ticker = ticker.upper()
    svc = service or build_local_api_service(root=root_path, use_voyage=False)
    steps: list[WorkflowStep] = []

    readiness = build_readiness_report(svc.chunks, svc.retriever.embeddings, tickers=[ticker], root=root_path)
    readiness_path = write_json_report(readiness.to_dict(), artifacts_dir / "phase6_readiness.json")
    steps.append(
        WorkflowStep(
            name="local_cache_readiness",
            status=_step_status(readiness.status),
            message=f"{readiness.chunk_count} chunks, {readiness.embedding_count} embeddings inspected.",
            details={"issue_count": len(readiness.issues), "status": readiness.status},
        )
    )

    query_payload = svc.query(QueryRequest(question=query, ticker=ticker, top_k=5, per_subquery_k=5))
    evidence_quality = build_evidence_quality_report(query_payload)
    query_path = write_json_report(query_payload, artifacts_dir / "phase6_query_smoke.json")
    evidence_path = write_json_report(evidence_quality.to_dict(), artifacts_dir / "phase6_evidence_quality.json")
    steps.append(
        WorkflowStep(
            name="query_smoke",
            status=_step_status(evidence_quality.status),
            message=f"{evidence_quality.result_count} retrieved chunks for {ticker}.",
            details={
                "question": query,
                "evidence_quality_status": evidence_quality.status,
                "issue_count": len(evidence_quality.issues),
            },
        )
    )

    evidence = evidence_rows(query_payload)
    coverage = coverage_rows(query_payload["coverage"])
    steps.append(
        WorkflowStep(
            name="workbench_smoke",
            status="pass" if evidence and coverage else "fail",
            message="Workbench helper rows rendered from query payload.",
            details={"evidence_rows": len(evidence), "coverage_rows": len(coverage)},
        )
    )

    differentiators = svc.differentiators(ticker=ticker)
    differentiator_path = write_json_report(differentiators, artifacts_dir / "phase6_differentiators.json")
    steps.append(
        WorkflowStep(
            name="differentiator_report",
            status="pass",
            message="Local differentiator payload built without provider calls.",
            details={
                "change_rows": len(change_rows(differentiators)),
                "language_signal_rows": len(language_signal_rows(differentiators)),
                "xbrl_status": differentiators.get("xbrl", {}).get("status", ""),
            },
        )
    )

    retrieved_by_query_id = _run_eval_queries(svc, ticker=ticker)
    eval_report = build_retrieval_eval_report(PHASE3_ROUTED_RETRIEVAL_FIXTURES, retrieved_by_query_id, k=5)
    eval_path = write_json_report(eval_report, artifacts_dir / "phase6_eval_report.json")
    steps.append(
        WorkflowStep(
            name="eval_report",
            status="pass",
            message="Tiny offline retrieval eval report written.",
            details={
                "case_count": eval_report["case_count"],
                "labeled_case_count": eval_report["labeled_case_count"],
                "unlabeled_case_count": eval_report["unlabeled_case_count"],
            },
        )
    )

    artifact_paths = {
        "readiness": str(readiness_path),
        "query_smoke": str(query_path),
        "evidence_quality": str(evidence_path),
        "differentiators": str(differentiator_path),
        "eval_report": str(eval_path),
    }
    report = DemoWorkflowReport(
        status="fail" if any(step.status == "fail" for step in steps) else "warning"
        if any(step.status == "warning" for step in steps)
        else "pass",
        prerequisites=[
            "Local chunks under data/filings/chunks/",
            "Local vectors under data/vector_cache/",
            "Optional local companyfacts JSON under data/companyfacts/{TICKER}.json",
            "No SEC refetch, OpenAI, Anthropic, or paid transcript APIs are used.",
        ],
        steps=steps,
        artifact_paths=artifact_paths,
        next_actions=_next_actions(readiness.to_dict(), evidence_quality.to_dict()),
    )
    write_json_report(report.to_dict(), artifacts_dir / "phase6_demo_workflow.json")
    return report


def _run_eval_queries(service: LocalRagApiService, *, ticker: str) -> dict[str, list[str]]:
    retrieved: dict[str, list[str]] = {}
    for case in PHASE3_ROUTED_RETRIEVAL_FIXTURES:
        try:
            payload = service.query(QueryRequest(question=case.question, ticker=ticker, top_k=5, per_subquery_k=5))
        except LocalApiError:
            retrieved[case.query_id] = []
        else:
            retrieved[case.query_id] = [str(result.get("chunk_id", "")) for result in payload.get("results", [])]
    return retrieved


def _step_status(status: str) -> str:
    if status in {"ready", "pass"}:
        return "pass"
    if status == "fail":
        return "fail"
    return "warning"


def _next_actions(readiness: dict[str, Any], evidence_quality: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if readiness.get("missing_embedding_count"):
        actions.append("Regenerate missing Voyage vector-cache embeddings before a live demo.")
    if readiness.get("missing_item_metadata_count"):
        actions.append("Regenerate SEC-aware chunks to improve item-level filtering and change detection.")
    if any(not available for available in readiness.get("companyfacts_available", {}).values()):
        actions.append("Add local companyfacts JSON when XBRL talking points matter.")
    if evidence_quality.get("missing_url_count") or evidence_quality.get("invalid_citation_count"):
        actions.append("Fix citation/source metadata before showing evidence as demo-ready.")
    if not actions:
        actions.append("Use the workbench command to walk through retrieval, citations, and differentiators.")
    return actions
