"""Evidence payload quality checks with no LLM/provider calls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceQualityIssue:
    code: str
    severity: str
    message: str
    chunk_id: str = ""
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceQualityReport:
    status: str
    result_count: int
    duplicate_chunk_ids: list[str]
    missing_url_count: int
    missing_accession_count: int
    missing_date_count: int
    missing_ticker_count: int
    missing_parent_context_count: int
    invalid_citation_count: int
    issues: list[EvidenceQualityIssue]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        return payload


def build_evidence_quality_report(query_payload: dict[str, Any]) -> EvidenceQualityReport:
    """Validate deterministic citation and metadata quality for retrieved chunks."""

    results = list(query_payload.get("results", []))
    labels = {str(result.get("citation_label", "")) for result in results if result.get("citation_label")}
    accepted = list(query_payload.get("citations", {}).get("accepted", []))
    rejected = [str(label) for label in query_payload.get("citations", {}).get("rejected", [])]
    accepted_labels = {str(citation.get("label", "")) for citation in accepted}
    accepted_labels.update(str(citation.get("citation_label", "")) for citation in accepted)
    accepted_labels.discard("")
    invalid_labels = sorted((accepted_labels | set(rejected)) - labels)

    duplicate_ids = _duplicates(str(result.get("chunk_id", "")) for result in results if result.get("chunk_id"))
    missing_url = _missing_results(results, result_key="source_url")
    missing_accession = _missing_metadata(results, "accession_number")
    missing_date = _missing_metadata(results, "filing_date")
    missing_ticker = _missing_metadata(results, "ticker")
    missing_parent = [
        str(result.get("chunk_id", ""))
        for result in results
        if not _has_parent_context(result.get("parent_context"))
    ]

    issues: list[EvidenceQualityIssue] = []
    for chunk_id in duplicate_ids[:5]:
        issues.append(
            EvidenceQualityIssue(
                code="duplicate_chunk_id",
                severity="warning",
                message="The same chunk appears more than once in a query result.",
                chunk_id=chunk_id,
            )
        )
    _add_count_issue(issues, "missing_source_url", "fail", "Retrieved chunks must include source URLs.", missing_url)
    _add_count_issue(
        issues,
        "missing_accession",
        "warning",
        "Retrieved chunks should include SEC accession metadata.",
        missing_accession,
    )
    _add_count_issue(
        issues,
        "missing_filing_date",
        "warning",
        "Retrieved chunks should include filing dates.",
        missing_date,
    )
    _add_count_issue(
        issues,
        "missing_ticker",
        "warning",
        "Retrieved chunks should include ticker metadata.",
        missing_ticker,
    )
    _add_count_issue(
        issues,
        "missing_parent_context",
        "info",
        "Parent context is not available for some retrieved chunks.",
        missing_parent,
    )
    if invalid_labels:
        issues.append(
            EvidenceQualityIssue(
                code="invalid_citation_label",
                severity="fail",
                message="Citation validation references labels that are not in retrieved results.",
                examples=invalid_labels[:5],
            )
        )

    status = "fail" if any(issue.severity == "fail" for issue in issues) else "warning" if issues else "pass"
    return EvidenceQualityReport(
        status=status,
        result_count=len(results),
        duplicate_chunk_ids=duplicate_ids,
        missing_url_count=len(missing_url),
        missing_accession_count=len(missing_accession),
        missing_date_count=len(missing_date),
        missing_ticker_count=len(missing_ticker),
        missing_parent_context_count=len(missing_parent),
        invalid_citation_count=len(invalid_labels),
        issues=issues,
    )


def _duplicates(values: object) -> list[str]:
    seen: set[str] = set()
    duplicated: set[str] = set()
    for value in values:
        if value in seen:
            duplicated.add(value)
        seen.add(value)
    return sorted(duplicated)


def _missing_results(results: list[dict[str, Any]], *, result_key: str) -> list[str]:
    return [str(result.get("chunk_id", "")) for result in results if not str(result.get(result_key, "")).strip()]


def _missing_metadata(results: list[dict[str, Any]], key: str) -> list[str]:
    return [
        str(result.get("chunk_id", ""))
        for result in results
        if not str(result.get("metadata", {}).get(key, "")).strip()
    ]


def _has_parent_context(parent_context: Any) -> bool:
    if not isinstance(parent_context, dict):
        return False
    if parent_context.get("context_text"):
        return True
    return bool(parent_context.get("context_chunk_ids"))


def _add_count_issue(
    issues: list[EvidenceQualityIssue],
    code: str,
    severity: str,
    message: str,
    chunk_ids: list[str],
) -> None:
    if chunk_ids:
        issues.append(
            EvidenceQualityIssue(
                code=code,
                severity=severity,
                message=message,
                examples=chunk_ids[:5],
            )
        )
