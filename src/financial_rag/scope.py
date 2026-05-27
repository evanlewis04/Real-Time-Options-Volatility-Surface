"""Shared constants for the financial filings RAG build-out."""

from __future__ import annotations

from enum import Enum


class FilingDocumentType(str, Enum):
    """Document categories the RAG pipeline should normalize."""

    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"
    EX_99 = "EX-99"
    PRESS_RELEASE = "PRESS_RELEASE"
    CFO_COMMENTARY = "CFO_COMMENTARY"
    PREPARED_REMARKS = "PREPARED_REMARKS"
    TRANSCRIPT = "TRANSCRIPT"


class QueryType(str, Enum):
    """Analyst query classes supported by the target product."""

    SINGLE_DOC_LOOKUP = "single_doc_lookup"
    TEMPORAL = "temporal"
    CROSS_SOURCE = "cross_source"
    CROSS_COMPANY = "cross_company"
    SPEAKER_SPECIFIC = "speaker_specific"
    MARKET_CONTEXT = "market_context"


DOCUMENT_TYPES: tuple[FilingDocumentType, ...] = tuple(FilingDocumentType)
CORE_QUERY_TYPES: tuple[QueryType, ...] = (
    QueryType.SINGLE_DOC_LOOKUP,
    QueryType.TEMPORAL,
    QueryType.CROSS_SOURCE,
    QueryType.CROSS_COMPANY,
    QueryType.SPEAKER_SPECIFIC,
)
