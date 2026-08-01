"""Shared Phase 1 data records for filings, chunks, and embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FilingMetadata:
    """Source metadata retained with every downloaded SEC document."""

    document_id: str
    ticker: str
    cik: str
    company_name: str
    accession_number: str
    form_type: str
    filing_date: str
    report_date: str
    source_url: str
    local_path: str
    document_role: str
    exhibit_type: str = ""
    document_name: str = ""
    description: str = ""
    content_hash: str = ""
    # Point-in-time spine (Phase 1): filed_at = EDGAR acceptance datetime (ISO-8601
    # UTC, the moment the filing became public); period_end = the reporting period
    # end (= report_date). Empty string means unresolved — never zero-filled/faked.
    filed_at: str = ""
    period_end: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentChunk:
    """Deterministic text chunk plus citation metadata."""

    chunk_id: str
    document_id: str
    ticker: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: str
    source_url: str
    local_path: str
    document_role: str
    exhibit_type: str
    chunk_text: str
    start_offset: int
    end_offset: int
    token_count: int
    section_path: str = ""
    item_number: str = ""
    speaker_name: str = ""
    speaker_role: str = ""
    # Point-in-time spine (Phase 1): threaded down from FilingMetadata so the record
    # retrieval sees carries acceptance datetime (filed_at) + period_end. Empty means
    # unresolved, never faked. See models.FilingMetadata.
    filed_at: str = ""
    period_end: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingRecord:
    """Cached embedding vector and source metadata."""

    chunk_id: str
    document_id: str
    model: str
    embedding: list[float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
