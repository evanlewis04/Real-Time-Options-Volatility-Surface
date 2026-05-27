"""Phase 0 source viability probe for the financial filings RAG track.

This script intentionally stops at source inspection. It does not download a
corpus, create chunks, call embedding providers, or invoke LLMs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


DEFAULT_TICKERS: tuple[str, ...] = ("NVDA", "MSFT", "AAPL", "AMD", "JPM")
DEFAULT_OUTPUT_PATH = Path("docs/financial-rag-phase0-findings.md")
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
RECENT_8K_LIMIT = 5
REQUEST_TIMEOUT_SECONDS = 30
PLACEHOLDER_MARKERS = ("your_", "example.com", "replace_me", "changeme")


@dataclass(frozen=True)
class FilingRecord:
    """Minimal SEC filing metadata needed for the Phase 0 report."""

    form: str
    accession_number: str
    filing_date: str
    report_date: str
    primary_document: str
    description: str

    @property
    def accession_directory(self) -> str:
        return accession_directory(self.accession_number)


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
class TickerProbeResult:
    """Per-company SEC probe result."""

    ticker: str
    company_name: str
    cik: str | None
    cik_status: str
    latest_10k: FilingRecord | None
    latest_10q: FilingRecord | None
    recent_8ks: tuple[FilingRecord, ...]
    ex99_by_accession: dict[str, tuple[ExhibitRecord, ...]]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlphaVantageResult:
    """Alpha Vantage transcript endpoint probe result."""

    status: str
    detail: str
    tested_symbol: str | None = None
    tested_quarter: str | None = None


class RateLimitedSession:
    """Small requests wrapper that enforces SEC-friendly spacing."""

    def __init__(self, *, headers: dict[str, str], delay_seconds: float) -> None:
        self._session = requests.Session()
        self._session.headers.update(headers)
        self._delay_seconds = delay_seconds
        self._last_request_at = 0.0

    def get_json(self, url: str, *, params: dict[str, str] | None = None) -> Any:
        response = self._get(url, params=params)
        return response.json()

    def get_text(self, url: str) -> str:
        return self._get(url).text

    def _get(self, url: str, *, params: dict[str, str] | None = None) -> requests.Response:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._delay_seconds:
            time.sleep(self._delay_seconds - elapsed)
        response = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        self._last_request_at = time.monotonic()
        response.raise_for_status()
        return response


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker input to SEC metadata casing."""

    return ticker.strip().upper()


def is_configured_secret(value: str | None) -> bool:
    """Return true for values that are present and not obvious placeholders."""

    if not value:
        return False
    lowered = value.strip().lower()
    return bool(lowered) and not any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def cik_padded(cik: int | str) -> str:
    """Return a SEC submissions-compatible ten-digit CIK string."""

    return str(cik).strip().zfill(10)


def cik_archive_value(cik: int | str) -> str:
    """Return the CIK form SEC Archives URLs use inside /data/."""

    return str(int(str(cik).strip()))


def accession_directory(accession_number: str) -> str:
    """Return the accession directory name used in SEC Archives URLs."""

    return accession_number.replace("-", "")


def sec_archive_base_url(cik: int | str, accession_number: str) -> str:
    """Build the SEC Archives filing directory URL."""

    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_archive_value(cik)}/"
        f"{accession_directory(accession_number)}/"
    )


def filing_index_url(cik: int | str, accession_number: str) -> str:
    """Build the human-readable SEC filing index page URL."""

    return f"{sec_archive_base_url(cik, accession_number)}{accession_number}-index.html"


def filing_primary_document_url(cik: int | str, filing: FilingRecord) -> str:
    """Build the URL for a filing's primary document."""

    return urljoin(sec_archive_base_url(cik, filing.accession_number), filing.primary_document)


def company_ticker_map(company_tickers_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert SEC company ticker payload to a ticker-keyed mapping."""

    mapped: dict[str, dict[str, Any]] = {}
    for entry in company_tickers_payload.values():
        ticker = normalize_ticker(str(entry.get("ticker", "")))
        if ticker:
            mapped[ticker] = entry
    return mapped


def recent_filing_records(submissions_payload: dict[str, Any]) -> tuple[FilingRecord, ...]:
    """Extract recent filing rows from a SEC submissions payload."""

    recent = submissions_payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_documents = recent.get("primaryDocument", [])
    descriptions = recent.get("primaryDocDescription", [])

    records: list[FilingRecord] = []
    for index, form in enumerate(forms):
        records.append(
            FilingRecord(
                form=str(form),
                accession_number=_list_get(accessions, index),
                filing_date=_list_get(filing_dates, index),
                report_date=_list_get(report_dates, index),
                primary_document=_list_get(primary_documents, index),
                description=_list_get(descriptions, index),
            )
        )
    return tuple(records)


def latest_filing(records: tuple[FilingRecord, ...], form: str) -> FilingRecord | None:
    """Return the latest exact-form filing from SEC recent filings."""

    for record in records:
        if record.form == form:
            return record
    return None


def recent_filings(
    records: tuple[FilingRecord, ...], form: str, *, limit: int
) -> tuple[FilingRecord, ...]:
    """Return recent exact-form filings up to the requested limit."""

    return tuple(record for record in records if record.form == form)[:limit]


def parse_filing_index_exhibits(html: str, *, base_url: str) -> tuple[ExhibitRecord, ...]:
    """Parse EX-99 rows from a SEC filing index HTML page."""

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
    """Return true if a filing attachment looks like an EX-99 exhibit."""

    normalized_type = exhibit_type.upper()
    if normalized_type.startswith("EX-99") or normalized_type.startswith("EX99"):
        return True

    document_lower = document.lower()
    exhibit_document = "ex99" in document_lower or "ex-99" in document_lower
    document_like = document_lower.endswith((".htm", ".html", ".txt", ".pdf"))
    description_mentions_exhibit = "EX-99" in description.upper() or "EX99" in description.upper()
    return exhibit_document and document_like and description_mentions_exhibit


def summarize_alpha_vantage_response(payload: dict[str, Any]) -> str:
    """Summarize Alpha Vantage transcript endpoint behavior without storing text."""

    if "Error Message" in payload:
        return f"error: {payload['Error Message']}"
    if "Information" in payload:
        return f"information: {payload['Information']}"
    if "Note" in payload:
        return f"rate_limit_or_note: {payload['Note']}"
    if not payload:
        return "empty response"

    transcript = payload.get("transcript")
    if isinstance(transcript, list):
        speaker_turns = len(transcript)
        sample_keys = sorted(transcript[0].keys()) if transcript else []
        return f"ok: transcript list with {speaker_turns} turns; sample keys: {sample_keys}"
    if isinstance(transcript, str):
        return f"ok: transcript text blob with {len(transcript)} characters"

    keys = sorted(payload.keys())
    return f"response schema not recognized; top-level keys: {keys}"


def probe_sec_sources(
    *,
    tickers: tuple[str, ...],
    user_agent: str,
    delay_seconds: float,
) -> tuple[TickerProbeResult, ...]:
    """Probe SEC ticker metadata, submissions, and EX-99 index coverage."""

    session = RateLimitedSession(
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        },
        delay_seconds=delay_seconds,
    )
    company_payload = session.get_json(SEC_COMPANY_TICKERS_URL)
    ticker_map = company_ticker_map(company_payload)
    results: list[TickerProbeResult] = []

    for ticker in tickers:
        normalized_ticker = normalize_ticker(ticker)
        metadata = ticker_map.get(normalized_ticker)
        if not metadata:
            results.append(
                TickerProbeResult(
                    ticker=normalized_ticker,
                    company_name="",
                    cik=None,
                    cik_status="missing",
                    latest_10k=None,
                    latest_10q=None,
                    recent_8ks=(),
                    ex99_by_accession={},
                    errors=("Ticker was not found in SEC company_tickers.json.",),
                )
            )
            continue

        cik = cik_padded(metadata["cik_str"])
        company_name = str(metadata.get("title", ""))
        errors: list[str] = []
        latest_10k = None
        latest_10q = None
        recent_8ks_for_ticker: tuple[FilingRecord, ...] = ()
        ex99_by_accession: dict[str, tuple[ExhibitRecord, ...]] = {}

        try:
            submissions = session.get_json(SEC_SUBMISSIONS_URL.format(cik=cik))
            records = recent_filing_records(submissions)
            latest_10k = latest_filing(records, "10-K")
            latest_10q = latest_filing(records, "10-Q")
            recent_8ks_for_ticker = recent_filings(records, "8-K", limit=RECENT_8K_LIMIT)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
            errors.append(f"SEC submissions fetch failed: {exc}")

        for filing in recent_8ks_for_ticker:
            try:
                base_url = sec_archive_base_url(cik, filing.accession_number)
                index_html = session.get_text(filing_index_url(cik, filing.accession_number))
                ex99_by_accession[filing.accession_number] = parse_filing_index_exhibits(
                    index_html,
                    base_url=base_url,
                )
            except requests.RequestException as exc:
                errors.append(f"{filing.accession_number} index fetch failed: {exc}")
                ex99_by_accession[filing.accession_number] = ()

        results.append(
            TickerProbeResult(
                ticker=normalized_ticker,
                company_name=company_name,
                cik=cik,
                cik_status="resolved",
                latest_10k=latest_10k,
                latest_10q=latest_10q,
                recent_8ks=recent_8ks_for_ticker,
                ex99_by_accession=ex99_by_accession,
                errors=tuple(errors),
            )
        )

    return tuple(results)


def probe_alpha_vantage(
    *,
    api_key: str | None,
    symbol: str,
    quarter: str,
    delay_seconds: float,
) -> AlphaVantageResult:
    """Probe the Alpha Vantage earnings-call transcript endpoint if configured."""

    if not is_configured_secret(api_key):
        return AlphaVantageResult(
            status="skipped",
            detail=(
                "ALPHA_VANTAGE_API_KEY is not configured. Add it to .env, then rerun "
                "this script to test function=EARNINGS_CALL_TRANSCRIPT."
            ),
        )

    session = RateLimitedSession(
        headers={"User-Agent": "financial-rag-phase0-probe"},
        delay_seconds=delay_seconds,
    )
    try:
        payload = session.get_json(
            ALPHA_VANTAGE_URL,
            params={
                "function": "EARNINGS_CALL_TRANSCRIPT",
                "symbol": normalize_ticker(symbol),
                "quarter": quarter,
                "apikey": api_key or "",
            },
        )
    except (requests.RequestException, json.JSONDecodeError) as exc:
        return AlphaVantageResult(
            status="error",
            detail=f"Alpha Vantage request failed: {exc}",
            tested_symbol=normalize_ticker(symbol),
            tested_quarter=quarter,
        )

    summary = summarize_alpha_vantage_response(payload)
    status = "ok" if summary.startswith("ok:") else "needs_review"
    return AlphaVantageResult(
        status=status,
        detail=summary,
        tested_symbol=normalize_ticker(symbol),
        tested_quarter=quarter,
    )


def render_markdown_report(
    *,
    run_at: datetime,
    tickers: tuple[str, ...],
    sec_results: tuple[TickerProbeResult, ...],
    alpha_result: AlphaVantageResult,
) -> str:
    """Render a human-readable Phase 0 findings report."""

    lines: list[str] = [
        "# Financial RAG Phase 0 Findings",
        "",
        f"- Date run: {run_at.astimezone(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- Tickers checked: {', '.join(tickers)}",
        "- Scope: SEC ticker metadata, SEC submissions, recent 8-K filing indexes, "
        "EX-99 exhibit coverage, and Alpha Vantage transcript endpoint availability.",
        "- Out of scope: embeddings, vector databases, chunking, retrieval, LLM calls, "
        "answer synthesis, and downloaded filing corpus storage.",
        "",
        "## SEC CIK Resolution",
        "",
        "| Ticker | Company | CIK | Status |",
        "| --- | --- | --- | --- |",
    ]

    for result in sec_results:
        lines.append(
            "| "
            f"{result.ticker} | {escape_table(result.company_name)} | "
            f"{result.cik or ''} | {result.cik_status} |"
        )

    lines.extend(
        [
            "",
            "## SEC Filing Coverage",
            "",
            "| Ticker | Latest 10-K | Latest 10-Q | Recent 8-K count | Recent 8-K accessions |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )

    for result in sec_results:
        latest_10k = filing_link(result.cik, result.latest_10k)
        latest_10q = filing_link(result.cik, result.latest_10q)
        eight_k_links = ", ".join(filing_link(result.cik, filing) for filing in result.recent_8ks)
        lines.append(
            "| "
            f"{result.ticker} | {latest_10k} | {latest_10q} | "
            f"{len(result.recent_8ks)} | {eight_k_links or 'None found'} |"
        )

    lines.extend(
        [
            "",
            "## EX-99 Exhibit Coverage From Recent 8-K Index Pages",
            "",
            "| Ticker | 8-K accession | Filing date | EX-99 count | Exhibit filenames and descriptions |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )

    total_8ks = 0
    total_8ks_with_ex99 = 0
    total_ex99 = 0
    for result in sec_results:
        for filing in result.recent_8ks:
            total_8ks += 1
            exhibits = result.ex99_by_accession.get(filing.accession_number, ())
            if exhibits:
                total_8ks_with_ex99 += 1
            total_ex99 += len(exhibits)
            exhibit_text = "; ".join(exhibit_link(exhibit) for exhibit in exhibits)
            lines.append(
                "| "
                f"{result.ticker} | {filing_link(result.cik, filing)} | "
                f"{filing.filing_date} | {len(exhibits)} | "
                f"{exhibit_text or 'No EX-99 exhibit found'} |"
            )

    lines.extend(
        [
            "",
            "## Alpha Vantage Transcript Probe",
            "",
            f"- Status: {alpha_result.status}",
            f"- Tested symbol: {alpha_result.tested_symbol or 'not run'}",
            f"- Tested quarter: {alpha_result.tested_quarter or 'not run'}",
            f"- Result: {alpha_result.detail}",
            "",
            "## Recommendation",
            "",
            recommendation_text(
                total_8ks=total_8ks,
                total_8ks_with_ex99=total_8ks_with_ex99,
                total_ex99=total_ex99,
                alpha_result=alpha_result,
            ),
        ]
    )

    errors = [error for result in sec_results for error in result.errors]
    if errors:
        lines.extend(["", "## Probe Errors", ""])
        lines.extend(f"- {error}" for error in errors)

    return "\n".join(lines) + "\n"


def recommendation_text(
    *,
    total_8ks: int,
    total_8ks_with_ex99: int,
    total_ex99: int,
    alpha_result: AlphaVantageResult,
) -> str:
    """Summarize whether free sources are enough for the intended v1."""

    ex99_rate = (total_8ks_with_ex99 / total_8ks) if total_8ks else 0.0
    if alpha_result.status == "ok":
        transcript_clause = (
            "Alpha Vantage returned a usable transcript-shaped response, so it remains "
            "a viable free transcript candidate pending broader coverage checks."
        )
    elif alpha_result.status == "skipped":
        transcript_clause = (
            "Alpha Vantage was skipped because no key was configured, so true "
            "earnings-call transcript coverage is still unproven."
        )
    else:
        transcript_clause = (
            "Alpha Vantage did not return a clearly usable transcript response in this "
            "probe, so it needs follow-up before becoming part of v1."
        )

    if ex99_rate >= 0.5 and total_ex99:
        sec_clause = (
            "SEC EDGAR looks strong enough for the free filing backbone and for "
            "earnings-release-adjacent EX-99 material."
        )
    else:
        sec_clause = (
            "SEC EDGAR is still the filing backbone, but recent EX-99 coverage is "
            "uneven enough that the app should surface coverage gaps prominently."
        )

    paid_clause = (
        "A paid transcript source is not required to begin free-only filing RAG, "
        "but it is still likely needed for complete speaker-aware Q&A transcript "
        "workflows unless Alpha Vantage coverage proves broad and cache-friendly."
    )
    return f"{sec_clause} {transcript_clause} {paid_clause}"


def filing_link(cik: str | None, filing: FilingRecord | None) -> str:
    """Render a Markdown filing link for report tables."""

    if not cik or not filing:
        return "None found"
    label = f"{filing.form} {filing.filing_date} ({filing.accession_number})"
    return f"[{label}]({filing_primary_document_url(cik, filing)})"


def exhibit_link(exhibit: ExhibitRecord) -> str:
    """Render a compact Markdown EX-99 exhibit link."""

    description = exhibit_description(exhibit)
    return f"[{exhibit.document}]({exhibit.url}) - {escape_table(description)}"


def exhibit_description(exhibit: ExhibitRecord) -> str:
    """Return the SEC description plus a light filename-based classification."""

    description = exhibit.description or exhibit.exhibit_type or "EX-99 exhibit"
    inferred = infer_exhibit_kind(exhibit.document, description)
    if inferred and inferred.lower() not in description.lower():
        return f"{description}; inferred: {inferred}"
    return description


def infer_exhibit_kind(document: str, description: str) -> str:
    """Infer a rough exhibit kind from SEC filename and description text."""

    text = f"{document} {description}".lower()
    if "cfo" in text or "commentary" in text:
        return "CFO commentary"
    if "slide" in text or "presentation" in text:
        return "presentation slides"
    if "press" in text or "pr." in text or "pr_" in text or document.lower().endswith("pr.htm"):
        return "press release"
    return ""


def escape_table(value: str) -> str:
    """Escape Markdown table separator characters."""

    return value.replace("|", "\\|")


def _list_get(values: list[Any], index: int) -> str:
    if index >= len(values):
        return ""
    return str(values[index])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Phase 0 financial RAG source viability.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="Tickers to probe. Defaults to the Phase 0 demo set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Markdown report path.",
    )
    parser.add_argument(
        "--sec-delay",
        type=float,
        default=0.25,
        help="Minimum seconds between SEC requests.",
    )
    parser.add_argument(
        "--alpha-symbol",
        default="NVDA",
        help="Ticker used for the Alpha Vantage transcript smoke test.",
    )
    parser.add_argument(
        "--alpha-quarter",
        default="2025Q3",
        help="Fiscal quarter used for the Alpha Vantage transcript smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()
    tickers = tuple(normalize_ticker(ticker) for ticker in args.tickers)
    user_agent = os.getenv("SEC_USER_AGENT")
    if not is_configured_secret(user_agent):
        raise SystemExit(
            "SEC_USER_AGENT must be configured in .env before probing SEC endpoints. "
            "Use a real contact string such as 'Your Name your.email@example.com'."
        )

    sec_results = probe_sec_sources(
        tickers=tickers,
        user_agent=user_agent,
        delay_seconds=max(args.sec_delay, 0.1),
    )
    alpha_result = probe_alpha_vantage(
        api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        symbol=args.alpha_symbol,
        quarter=args.alpha_quarter,
        delay_seconds=1.0,
    )
    report = render_markdown_report(
        run_at=datetime.now(tz=UTC),
        tickers=tickers,
        sec_results=sec_results,
        alpha_result=alpha_result,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote Phase 0 findings report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
