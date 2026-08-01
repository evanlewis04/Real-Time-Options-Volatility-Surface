"""Phase 1 Stage 1 — point-in-time metadata spine.

Covers acceptance-datetime parsing, its capture into FilingRecord, and the
threading of filed_at / period_end onto every chunk (including the record the
retriever actually sees via ``_chunk_metadata``).
"""

from src.financial_rag.chunking import chunk_sec_document
from src.financial_rag.chunking.simple import chunk_document
from src.financial_rag.ingestion.sec_client import (
    parse_acceptance_datetime,
    recent_filing_records,
)
from src.financial_rag.models import FilingMetadata
from src.financial_rag.retrieval.local_dense import _chunk_metadata


def _metadata(**overrides: str) -> FilingMetadata:
    base = dict(
        document_id="NVDA-doc",
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA CORP",
        accession_number="0001045810-26-000051",
        form_type="10-Q",
        filing_date="2026-05-20",
        report_date="2026-04-26",
        source_url="https://www.sec.gov/Archives/doc.htm",
        local_path="data/filings/raw/NVDA/doc.htm",
        document_role="primary",
        exhibit_type="",
        document_name="doc.htm",
        description="10-Q",
        content_hash="abc123",
        filed_at="2026-05-20T20:31:12+00:00",
        period_end="2026-04-26",
    )
    base.update(overrides)
    return FilingMetadata(**base)


def test_parse_acceptance_datetime_normalizes_to_utc() -> None:
    # EDGAR's canonical Z-suffixed UTC form.
    assert parse_acceptance_datetime("2026-02-25T16:31:12.000Z") == "2026-02-25T16:31:12+00:00"
    # An explicit offset is converted to UTC.
    assert parse_acceptance_datetime("2026-02-25T16:31:12-05:00") == "2026-02-25T21:31:12+00:00"
    # A naive value is assumed UTC (matches EDGAR's Z convention).
    assert parse_acceptance_datetime("2026-02-25T16:31:12") == "2026-02-25T16:31:12+00:00"
    # Fractional seconds are preserved.
    assert parse_acceptance_datetime("2026-02-25T16:31:12.500Z") == "2026-02-25T16:31:12.500000+00:00"


def test_parse_acceptance_datetime_flags_unresolvable_as_empty() -> None:
    # Data-honesty guardrail: missing/garbage is flagged empty, never faked.
    assert parse_acceptance_datetime("") == ""
    assert parse_acceptance_datetime("   ") == ""
    assert parse_acceptance_datetime("not-a-datetime") == ""
    assert parse_acceptance_datetime("2026-13-45T99:99:99Z") == ""


def test_recent_filing_records_captures_acceptance_datetime() -> None:
    payload = {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K"],
                "accessionNumber": ["0000034088-26-000045", "0001045810-26-000051"],
                "filingDate": ["2026-02-18", "2026-05-20"],
                "reportDate": ["2025-12-31", "2026-05-20"],
                "primaryDocument": ["xom-10k.htm", "nvda-8k.htm"],
                "primaryDocDescription": ["10-K", "8-K"],
                "acceptanceDateTime": [
                    "2026-02-18T17:02:05.000Z",
                    "",  # missing acceptance -> flagged empty
                ],
            }
        }
    }

    records = recent_filing_records(payload)

    assert records[0].acceptance_datetime == "2026-02-18T17:02:05+00:00"
    assert records[1].acceptance_datetime == ""  # not zero-filled


def test_simple_chunker_threads_point_in_time_fields() -> None:
    metadata = _metadata()
    text = "Alpha beta gamma delta. " * 200

    chunks = chunk_document(text, metadata, max_chars=300, overlap_chars=20)

    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.filed_at == "2026-05-20T20:31:12+00:00"
        assert chunk.period_end == "2026-04-26"
        # The record the retriever actually sees must carry the spine fields.
        surfaced = _chunk_metadata(chunk.to_dict())
        assert surfaced["filed_at"] == "2026-05-20T20:31:12+00:00"
        assert surfaced["period_end"] == "2026-04-26"


def test_sec_aware_chunker_threads_point_in_time_fields() -> None:
    metadata = _metadata(form_type="10-K")
    text = (
        "Item 1A. Risk Factors. " + ("Commodity prices may fluctuate materially. " * 80)
    )

    chunks = chunk_sec_document(text, metadata, max_chars=400, overlap_chars=40)

    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk.filed_at == "2026-05-20T20:31:12+00:00"
        assert chunk.period_end == "2026-04-26"
