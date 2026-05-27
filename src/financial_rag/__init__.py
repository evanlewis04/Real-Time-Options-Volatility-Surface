"""Financial filings RAG namespace.

The initial package is intentionally light. It gives the RAG implementation a
separate home beside the existing options-volatility modules while preserving
the current dashboard and quant package boundaries.
"""

from .scope import (
    CORE_QUERY_TYPES,
    DOCUMENT_TYPES,
    FilingDocumentType,
    QueryType,
)

__all__ = [
    "CORE_QUERY_TYPES",
    "DOCUMENT_TYPES",
    "FilingDocumentType",
    "QueryType",
]
