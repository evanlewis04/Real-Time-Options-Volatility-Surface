"""Local cache coverage summaries for filings retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.financial_rag.retrieval import LocalChunkRecord


@dataclass(frozen=True)
class TickerCoverage:
    ticker: str
    chunk_count: int
    ex99_chunk_count: int = 0
    has_press_release: bool = False
    has_cfo_commentary: bool = False
    has_prepared_remarks: bool = False
    form_types: list[str] = field(default_factory=list)
    accessions: list[str] = field(default_factory=list)
    filing_dates: list[str] = field(default_factory=list)
    document_roles: list[str] = field(default_factory=list)
    exhibit_types: list[str] = field(default_factory=list)
    item_numbers: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CoverageReport:
    tickers: dict[str, TickerCoverage]


def build_coverage_report(
    chunks: list[LocalChunkRecord],
    *,
    tickers: list[str] | None = None,
) -> CoverageReport:
    """Summarize what is locally cached, including honest EX-99 gaps."""

    requested = {ticker.upper() for ticker in tickers} if tickers else None
    grouped: dict[str, list[LocalChunkRecord]] = {}
    for chunk in chunks:
        ticker = str(chunk.metadata.get("ticker", "")).upper()
        if not ticker:
            continue
        if requested is not None and ticker not in requested:
            continue
        grouped.setdefault(ticker, []).append(chunk)

    if requested is not None:
        for ticker in requested:
            grouped.setdefault(ticker, [])

    return CoverageReport(
        tickers={ticker: _ticker_coverage(ticker, ticker_chunks) for ticker, ticker_chunks in sorted(grouped.items())}
    )


def _ticker_coverage(ticker: str, chunks: list[LocalChunkRecord]) -> TickerCoverage:
    form_types = _metadata_values(chunks, "form_type")
    document_roles = _metadata_values(chunks, "document_role")
    exhibit_types = _metadata_values(chunks, "exhibit_type")
    ex99_chunk_count = sum(1 for chunk in chunks if str(chunk.metadata.get("form_type", "")).upper() == "EX-99")
    has_press_release = "PRESS_RELEASE" in exhibit_types
    has_cfo_commentary = "CFO_COMMENTARY" in exhibit_types
    has_prepared_remarks = "PREPARED_REMARKS" in exhibit_types
    gaps: list[str] = []
    if not chunks:
        gaps.append("No local chunks cached.")
    if "EX-99" not in form_types:
        gaps.append("No cached EX-99 exhibits detected.")
    if not has_cfo_commentary:
        gaps.append("No cached CFO commentary detected.")
    if not has_prepared_remarks:
        gaps.append("No cached prepared remarks detected.")
    if not any((has_press_release, has_cfo_commentary, has_prepared_remarks)):
        gaps.append("No usable EX-99 narrative type detected.")

    return TickerCoverage(
        ticker=ticker,
        chunk_count=len(chunks),
        ex99_chunk_count=ex99_chunk_count,
        has_press_release=has_press_release,
        has_cfo_commentary=has_cfo_commentary,
        has_prepared_remarks=has_prepared_remarks,
        form_types=form_types,
        accessions=_metadata_values(chunks, "accession_number"),
        filing_dates=_metadata_values(chunks, "filing_date"),
        document_roles=document_roles,
        exhibit_types=exhibit_types,
        item_numbers=_metadata_values(chunks, "item_number"),
        speakers=_metadata_values(chunks, "speaker_name"),
        gaps=gaps,
    )


def _metadata_values(chunks: list[LocalChunkRecord], key: str) -> list[str]:
    values = {str(chunk.metadata.get(key, "")).strip() for chunk in chunks}
    return sorted(value for value in values if value)
