"""SEC EDGAR ingestion primitives for the Phase 1 smoke pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.financial_rag.parsing.exhibits import classify_ex99_exhibit


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_SEC_DELAY_SECONDS = 0.2


@dataclass(frozen=True)
class CompanyRecord:
    """SEC ticker map record."""

    ticker: str
    cik: str
    company_name: str


@dataclass(frozen=True)
class FilingRecord:
    """Minimal SEC filing metadata needed for Phase 1 ingestion."""

    form: str
    accession_number: str
    filing_date: str
    report_date: str
    primary_document: str
    description: str
    # EDGAR acceptance datetime, normalized to ISO-8601 UTC. This is the precise
    # "when did this become public" instant the point-in-time spine relies on.
    # Empty string when the source value is missing/unparseable (never faked).
    acceptance_datetime: str = ""


def parse_acceptance_datetime(raw: str) -> str:
    """Normalize an EDGAR ``acceptanceDateTime`` to an ISO-8601 UTC string.

    EDGAR's submissions payload reports acceptance as e.g. ``2026-02-25T16:31:12.000Z``.
    Returns ``""`` for a missing or unparseable value — a data-honesty guardrail: an
    unresolved acceptance instant is flagged as empty, never zero-filled or guessed.
    A naive datetime (no offset) is assumed UTC, matching EDGAR's ``Z`` convention.
    """

    text = (raw or "").strip()
    if not text:
        return ""
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class ExhibitRecord:
    """Attachment metadata parsed from an SEC filing index page."""

    sequence: str
    description: str
    document: str
    exhibit_type: str
    size: str
    url: str


@dataclass(frozen=True)
class FilingSelection:
    """The Phase 1 one-ticker filing set."""

    latest_10k: FilingRecord | None
    latest_10q: FilingRecord | None
    recent_8ks: tuple[FilingRecord, ...]


class SECClient:
    """Rate-limited SEC client with fair-access headers."""

    def __init__(
        self,
        *,
        user_agent: str,
        delay_seconds: float = DEFAULT_SEC_DELAY_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("SEC_USER_AGENT is required for SEC EDGAR requests.")
        self.delay_seconds = max(delay_seconds, 0.1)
        self._last_request_at = 0.0
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov",
            }
        )

    def get_json(self, url: str, *, params: dict[str, str] | None = None) -> Any:
        response = self._get(url, params=params)
        return response.json()

    def get_text(self, url: str) -> str:
        return self._get(url).text

    def get_bytes(self, url: str) -> bytes:
        return self._get(url).content

    def fetch_company_tickers(self) -> dict[str, CompanyRecord]:
        payload = self.get_json(SEC_COMPANY_TICKERS_URL)
        return company_ticker_map(payload)

    def lookup_company(self, ticker: str) -> CompanyRecord:
        normalized = normalize_ticker(ticker)
        company = self.fetch_company_tickers().get(normalized)
        if company is None:
            raise KeyError(f"{normalized} was not found in SEC company_tickers.json.")
        return company

    def fetch_company_submissions(self, cik: int | str) -> dict[str, Any]:
        return self.get_json(SEC_SUBMISSIONS_URL.format(cik=cik_padded(cik)))

    def fetch_filing_index(self, cik: int | str, accession_number: str) -> str:
        return self.get_text(filing_index_url(cik, accession_number))

    def fetch_document(self, url: str) -> bytes:
        return self.get_bytes(url)

    def _get(self, url: str, *, params: dict[str, str] | None = None) -> requests.Response:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        headers = {}
        if "data.sec.gov" in url:
            headers["Host"] = "data.sec.gov"
        response = self._session.get(
            url,
            params=params,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def cik_padded(cik: int | str) -> str:
    """Return a SEC submissions-compatible ten-digit CIK string."""

    return str(cik).strip().zfill(10)


def cik_archive_value(cik: int | str) -> str:
    """Return the CIK form SEC Archives URLs use inside `/data/`."""

    return str(int(str(cik).strip()))


def accession_directory(accession_number: str) -> str:
    """Return the accession directory name used in SEC Archives URLs."""

    return accession_number.replace("-", "")


def sec_archive_base_url(cik: int | str, accession_number: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_archive_value(cik)}/"
        f"{accession_directory(accession_number)}/"
    )


def filing_index_url(cik: int | str, accession_number: str) -> str:
    return f"{sec_archive_base_url(cik, accession_number)}{accession_number}-index.html"


def filing_primary_document_url(cik: int | str, filing: FilingRecord) -> str:
    return urljoin(sec_archive_base_url(cik, filing.accession_number), filing.primary_document)


def build_document_id(
    *,
    ticker: str,
    accession_number: str,
    document_role: str,
    document_name: str,
) -> str:
    """Build a stable, filesystem-safe document identifier."""

    raw = f"{normalize_ticker(ticker)}|{accession_number}|{document_role}|{document_name}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "-", document_name).strip("-").lower()[:48]
    return f"{normalize_ticker(ticker)}-{accession_directory(accession_number)}-{safe_name}-{digest}"


def company_ticker_map(company_tickers_payload: dict[str, Any]) -> dict[str, CompanyRecord]:
    mapped: dict[str, CompanyRecord] = {}
    for entry in company_tickers_payload.values():
        ticker = normalize_ticker(str(entry.get("ticker", "")))
        if ticker:
            mapped[ticker] = CompanyRecord(
                ticker=ticker,
                cik=cik_padded(entry.get("cik_str", "")),
                company_name=str(entry.get("title", "")),
            )
    return mapped


def recent_filing_records(submissions_payload: dict[str, Any]) -> tuple[FilingRecord, ...]:
    recent = submissions_payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    records: list[FilingRecord] = []
    for index, form in enumerate(forms):
        records.append(
            FilingRecord(
                form=str(form),
                accession_number=_list_get(recent.get("accessionNumber", []), index),
                filing_date=_list_get(recent.get("filingDate", []), index),
                report_date=_list_get(recent.get("reportDate", []), index),
                primary_document=_list_get(recent.get("primaryDocument", []), index),
                description=_list_get(recent.get("primaryDocDescription", []), index),
                acceptance_datetime=parse_acceptance_datetime(
                    _list_get(recent.get("acceptanceDateTime", []), index)
                ),
            )
        )
    return tuple(records)


def select_phase1_filings(
    records: tuple[FilingRecord, ...],
    *,
    recent_8k_limit: int = 5,
) -> FilingSelection:
    """Select latest 10-K, latest 10-Q, and recent exact-form 8-K filings."""

    return FilingSelection(
        latest_10k=latest_filing(records, "10-K"),
        latest_10q=latest_filing(records, "10-Q"),
        recent_8ks=recent_filings(records, "8-K", limit=recent_8k_limit),
    )


def latest_filing(records: tuple[FilingRecord, ...], form: str) -> FilingRecord | None:
    for record in records:
        if record.form == form:
            return record
    return None


def recent_filings(
    records: tuple[FilingRecord, ...], form: str, *, limit: int
) -> tuple[FilingRecord, ...]:
    return tuple(record for record in records if record.form == form)[:limit]


def parse_filing_index_exhibits(html: str, *, base_url: str) -> tuple[ExhibitRecord, ...]:
    """Parse EX-99 rows from an SEC filing index page."""

    soup = BeautifulSoup(html, "html.parser")
    exhibits: list[ExhibitRecord] = []
    for row in soup.select("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
        if len(cells) < 4:
            continue
        sequence = cells[0]
        description = cells[1] if len(cells) > 1 else ""
        document = cells[2] if len(cells) > 2 else ""
        exhibit_type = cells[3] if len(cells) > 3 else ""
        size = cells[4] if len(cells) > 4 else ""
        link = row.find("a", href=True)
        href = str(link["href"]) if link else document

        if is_ex99_exhibit(exhibit_type=exhibit_type, document=document, description=description):
            exhibits.append(
                ExhibitRecord(
                    sequence=sequence,
                    description=description,
                    document=document,
                    exhibit_type=exhibit_type,
                    size=size,
                    url=urljoin(base_url, href),
                )
            )
    return tuple(exhibits)


def is_ex99_exhibit(*, exhibit_type: str, document: str, description: str) -> bool:
    normalized_type = exhibit_type.upper()
    if normalized_type.startswith("EX-99") or normalized_type.startswith("EX99"):
        return True
    document_lower = document.lower()
    if not document_lower.endswith((".htm", ".html", ".txt", ".pdf")):
        return False
    description_mentions_exhibit = "EX-99" in description.upper() or "EX99" in description.upper()
    return ("ex99" in document_lower or "ex-99" in document_lower) and description_mentions_exhibit


def infer_exhibit_type(document: str, description: str) -> str:
    """Infer a rough exhibit role from filename and SEC description."""

    return classify_ex99_exhibit(filename=document, description=description)


def _list_get(values: list[Any], index: int) -> str:
    if index >= len(values):
        return ""
    return str(values[index])


def pretty_json(payload: Any) -> str:
    """Stable JSON for local manifests and debugging."""

    return json.dumps(payload, indent=2, sort_keys=True)
