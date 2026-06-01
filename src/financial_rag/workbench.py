"""Streamlit workbench helpers for filings evidence review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.financial_rag.api import QueryRequest


@dataclass(frozen=True)
class AnswerGate:
    """Whether an opt-in OpenAI answer may run over retrieved evidence."""

    allowed: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "reasons": list(self.reasons)}


def evaluate_answer_gate(
    query_payload: dict[str, Any],
    *,
    evidence_quality_status: str,
    openai_ready: bool,
    openai_issues: list[str] | None = None,
) -> AnswerGate:
    """Decide whether a generated answer may be offered for this query.

    The gate is intentionally conservative and evidence-first: an answer is only
    offered when there is grounded, quality-passing evidence and OpenAI is
    configured. It never hides coverage gaps; every block is explained in
    ``reasons`` so the workbench can show why an answer is unavailable.
    """

    reasons: list[str] = []
    if not query_payload.get("results"):
        reasons.append("No retrieved evidence is available to ground an answer.")
    if evidence_quality_status != "pass":
        reasons.append(f"Evidence quality is '{evidence_quality_status}', not 'pass'.")
    if not openai_ready:
        reasons.extend(openai_issues or ["OpenAI is not configured (set OPENAI_API_KEY and install openai)."])
    return AnswerGate(allowed=not reasons, reasons=reasons)


def answer_citation_rows(answer_payload: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten an answer's accepted citations into a source-audit table."""

    return [
        {
            "label": str(citation.get("label", "")),
            "ticker": str(citation.get("ticker", "")),
            "form_type": str(citation.get("form_type", "")),
            "filing_date": str(citation.get("filing_date", "")),
            "accession": str(citation.get("accession", "")),
            "source_url": str(citation.get("source_url", "")),
            "chunk_id": str(citation.get("chunk_id", "")),
        }
        for citation in answer_payload.get("accepted_citations", [])
    ]


def company_options(companies_payload: dict[str, Any]) -> list[str]:
    """Return cached, queryable tickers for the workbench ticker selector."""

    return [str(company.get("ticker", "")) for company in companies_payload.get("companies", []) if company.get("ticker")]


def evidence_rows(query_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten query results into an analyst-friendly evidence table."""

    rows: list[dict[str, Any]] = []
    for result in query_payload.get("results", []):
        metadata = result.get("metadata", {})
        rows.append(
            {
                "label": result.get("citation_label", ""),
                "score": result.get("score", 0.0),
                "ticker": metadata.get("ticker", ""),
                "form_type": metadata.get("form_type", ""),
                "filing_date": metadata.get("filing_date", ""),
                "accession": metadata.get("accession_number", ""),
                "role": metadata.get("document_role", ""),
                "exhibit_type": metadata.get("exhibit_type", ""),
                "item": metadata.get("item_number", ""),
                "speaker": metadata.get("speaker_name", ""),
                "source_url": result.get("source_url", ""),
                "chunk_id": result.get("chunk_id", ""),
                "subquery_id": result.get("subquery_id", ""),
            }
        )
    return rows


def coverage_rows(coverage_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten coverage payload for Streamlit display or CSV export."""

    rows: list[dict[str, Any]] = []
    for ticker, coverage in coverage_payload.get("tickers", {}).items():
        rows.append(
            {
                "ticker": ticker,
                "chunks": coverage.get("chunk_count", 0),
                "forms": ", ".join(coverage.get("form_types", [])),
                "accessions": ", ".join(coverage.get("accessions", [])),
                "dates": ", ".join(coverage.get("filing_dates", [])),
                "roles": ", ".join(coverage.get("document_roles", [])),
                "ex99_types": ", ".join(coverage.get("exhibit_types", [])),
                "items": ", ".join(coverage.get("item_numbers", [])),
                "speakers": ", ".join(coverage.get("speakers", [])),
                "gaps": " | ".join(coverage.get("gaps", [])),
            }
        )
    return rows


def rejected_citation_rows(query_payload: dict[str, Any]) -> list[dict[str, str]]:
    return [{"label": label} for label in query_payload.get("citations", {}).get("rejected", [])]


def change_rows(differentiator_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "change_id": record.get("change_id", ""),
            "ticker": record.get("ticker", ""),
            "item": record.get("item_number", ""),
            "type": record.get("change_type", ""),
            "previous_date": record.get("previous_filing_date", ""),
            "current_date": record.get("current_filing_date", ""),
            "previous_accession": record.get("previous_accession", ""),
            "current_accession": record.get("current_accession", ""),
            "previous_chunk": record.get("previous_chunk_id", ""),
            "current_chunk": record.get("current_chunk_id", ""),
        }
        for record in differentiator_payload.get("changes", [])
    ]


def language_signal_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "scope": record.get("scope", ""),
            "ticker": record.get("ticker", ""),
            "document_id": record.get("document_id", ""),
            "item": record.get("item_number", ""),
            "chunks": record.get("chunk_count", 0),
            "tokens": record.get("token_count", 0),
            "uncertainty": record.get("uncertainty_hits", 0),
            "risk": record.get("risk_hits", 0),
            "positive": record.get("positive_hits", 0),
            "negative": record.get("negative_hits", 0),
            "sentiment": record.get("sentiment_score", 0),
        }
        for record in payload.get("language_signals", [])
    ]


def readiness_issue_rows(readiness_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "severity": issue.get("severity", ""),
            "code": issue.get("code", ""),
            "ticker": issue.get("ticker", ""),
            "count": issue.get("count", 0),
            "message": issue.get("message", ""),
            "examples": ", ".join(issue.get("examples", [])),
        }
        for issue in readiness_payload.get("issues", [])
    ]


def evidence_quality_issue_rows(evidence_quality_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "severity": issue.get("severity", ""),
            "code": issue.get("code", ""),
            "chunk_id": issue.get("chunk_id", ""),
            "message": issue.get("message", ""),
            "examples": ", ".join(issue.get("examples", [])),
        }
        for issue in evidence_quality_payload.get("issues", [])
    ]


def default_query_request() -> QueryRequest:
    return QueryRequest()
