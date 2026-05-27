"""Streamlit workbench helpers for filings evidence review."""

from __future__ import annotations

from typing import Any

from src.financial_rag.api import QueryRequest


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
