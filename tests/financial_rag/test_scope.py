from src.financial_rag import (
    CORE_QUERY_TYPES,
    DOCUMENT_TYPES,
    FilingDocumentType,
    QueryType,
)


def test_core_query_types_match_initial_rag_goal() -> None:
    assert CORE_QUERY_TYPES == (
        QueryType.SINGLE_DOC_LOOKUP,
        QueryType.TEMPORAL,
        QueryType.CROSS_SOURCE,
        QueryType.CROSS_COMPANY,
        QueryType.SPEAKER_SPECIFIC,
    )


def test_document_types_include_sec_filings_and_exhibit_content() -> None:
    assert FilingDocumentType.TEN_K in DOCUMENT_TYPES
    assert FilingDocumentType.TEN_Q in DOCUMENT_TYPES
    assert FilingDocumentType.EIGHT_K in DOCUMENT_TYPES
    assert FilingDocumentType.EX_99 in DOCUMENT_TYPES
    assert FilingDocumentType.CFO_COMMENTARY in DOCUMENT_TYPES
