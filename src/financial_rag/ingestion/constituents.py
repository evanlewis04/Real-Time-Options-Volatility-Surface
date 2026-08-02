"""S&P 500 constituent membership: parse, resolve, and serialize ticker→CIK.

Phase 1 Stage 4 scales ingestion from one demo ticker to the S&P 500. The
constituent list is committed as a static CSV (``config/sp500_constituents.csv``)
so a fetch run never depends on a live ticker→CIK lookup. Resolution to CIK uses
the same ``company_tickers`` mapping ``sec_client`` already relies on; an
unresolvable ticker is flagged (returned in ``unresolved``), never guessed or
zero-filled (data-honesty guardrail).

**Membership is static current membership** (today's index), not point-in-time
historical membership. Companies added to or dropped from the index over time are
not tracked here — a survivorship-bias limitation that is acceptable for a corpus
snapshot but would need point-in-time membership for a historical backtest.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from src.financial_rag.ingestion.sec_client import (
    CompanyRecord,
    cik_padded,
    normalize_ticker,
)

CONSTITUENTS_HEADER = ("ticker", "cik", "company_name")


@dataclass(frozen=True)
class ResolutionResult:
    """Outcome of resolving a ticker list against the SEC ticker map."""

    resolved: tuple[CompanyRecord, ...]
    unresolved: tuple[str, ...]


def _data_lines(text: str) -> list[str]:
    """Return CSV lines, dropping blank and ``#``-comment lines.

    The committed CSV carries a provenance/limitation header as ``#`` comment
    lines; those must not reach ``csv.DictReader``.
    """

    return [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_constituents_csv(text: str) -> tuple[CompanyRecord, ...]:
    """Parse a committed constituents CSV into ``CompanyRecord``s.

    Rows without a ticker are skipped. A CIK is zero-padded to the 10-digit SEC
    form only when present; a missing CIK is left empty (flagged, never faked).
    """

    reader = csv.DictReader(_data_lines(text))
    records: list[CompanyRecord] = []
    for row in reader:
        ticker = normalize_ticker(str(row.get("ticker", "")))
        if not ticker:
            continue
        raw_cik = str(row.get("cik", "") or "").strip()
        records.append(
            CompanyRecord(
                ticker=ticker,
                cik=cik_padded(raw_cik) if raw_cik else "",
                company_name=str(row.get("company_name", "") or "").strip(),
            )
        )
    return tuple(records)


def resolve_constituents(
    tickers: Iterable[str],
    ticker_map: Mapping[str, CompanyRecord],
) -> ResolutionResult:
    """Resolve tickers to ``CompanyRecord``s via the SEC ``company_tickers`` map.

    Class-share tickers differ by punctuation between sources — index lists use a
    dot (``BRK.B``) while SEC uses a dash (``BRK-B``); both spellings are tried.
    Duplicates are collapsed on first sighting. An unresolvable ticker is returned
    in ``unresolved`` rather than dropped silently or mapped to a placeholder CIK.
    """

    resolved: list[CompanyRecord] = []
    unresolved: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        ticker = normalize_ticker(str(raw))
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        record = (
            ticker_map.get(ticker)
            or ticker_map.get(ticker.replace(".", "-"))
            or ticker_map.get(ticker.replace("-", "."))
        )
        if record is None:
            unresolved.append(ticker)
        else:
            resolved.append(record)
    return ResolutionResult(resolved=tuple(resolved), unresolved=tuple(unresolved))


def format_constituents_csv(
    records: Sequence[CompanyRecord],
    *,
    header_comment: Sequence[str] = (),
) -> str:
    """Serialize records to the committed CSV form, sorted by ticker.

    ``header_comment`` lines are emitted as ``#`` comments above the CSV header so
    provenance and the survivorship-bias limitation travel with the artifact.
    """

    out = io.StringIO()
    for line in header_comment:
        out.write(f"# {line}\n")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(CONSTITUENTS_HEADER)
    for record in sorted(records, key=lambda item: item.ticker):
        writer.writerow([record.ticker, record.cik, record.company_name])
    return out.getvalue()
