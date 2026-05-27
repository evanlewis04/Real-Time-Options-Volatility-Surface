"""Rule-based query routing for Phase 3."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.financial_rag.scope import QueryType


COMPANY_TICKERS: dict[str, str] = {
    "NVIDIA": "NVDA",
    "MICROSOFT": "MSFT",
    "APPLE": "AAPL",
    "AMD": "AMD",
    "ADVANCED MICRO DEVICES": "AMD",
    "INTEL": "INTC",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "META": "META",
    "AMAZON": "AMZN",
    "JPMORGAN": "JPM",
    "EXXON": "XOM",
    "EXXON MOBIL": "XOM",
}
KNOWN_TICKERS = {
    "NVDA",
    "MSFT",
    "AAPL",
    "AMD",
    "INTC",
    "GOOGL",
    "META",
    "AMZN",
    "JPM",
    "BAC",
    "GS",
    "WMT",
    "COST",
    "MCD",
    "UNH",
    "JNJ",
    "LLY",
    "XOM",
    "CVX",
    "CAT",
}


@dataclass(frozen=True)
class QueryFilters:
    tickers: list[str] = field(default_factory=list)
    company_names: list[str] = field(default_factory=list)
    form_types: list[str] = field(default_factory=list)
    time_window: str = ""
    last_n_quarters: int | None = None
    fiscal_periods: list[str] = field(default_factory=list)
    item_numbers: list[str] = field(default_factory=list)
    document_roles: list[str] = field(default_factory=list)
    exhibit_types: list[str] = field(default_factory=list)
    speaker_names: list[str] = field(default_factory=list)
    speaker_roles: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RoutedQuery:
    question: str
    query_type: QueryType
    filters: QueryFilters
    trace: dict[str, str | int | list[str]]


def route_query(question: str, *, default_ticker: str | None = None) -> RoutedQuery:
    """Classify and extract filters using deterministic text rules only."""

    filters = QueryFilters(
        tickers=_extract_tickers(question, default_ticker=default_ticker),
        company_names=_extract_company_names(question),
        form_types=_extract_form_types(question),
        time_window=_extract_time_window(question),
        last_n_quarters=_extract_last_n_quarters(question),
        fiscal_periods=_extract_fiscal_periods(question),
        item_numbers=_extract_item_numbers(question),
        document_roles=_extract_document_roles(question),
        exhibit_types=_extract_exhibit_types(question),
        speaker_names=_extract_speaker_names(question),
        speaker_roles=_extract_speaker_roles(question),
    )
    query_type = _classify_query(question, filters)
    trace: dict[str, str | int | list[str]] = {
        "router": "rule_based_v1",
        "query_type": query_type.value,
        "tickers": filters.tickers,
    }
    if filters.time_window:
        trace["time_window"] = filters.time_window
    return RoutedQuery(question=question, query_type=query_type, filters=filters, trace=trace)


def _classify_query(question: str, filters: QueryFilters) -> QueryType:
    lower = question.lower()
    if re.search(r"\b(iv|implied volatility|skew|options?|market context|expected move)\b", lower):
        return QueryType.MARKET_CONTEXT
    if len(filters.tickers) > 1 or len(filters.company_names) > 1:
        return QueryType.CROSS_COMPANY
    if any(term in lower for term in ("compare", "match", "versus", "vs")) and (
        filters.exhibit_types or len(filters.form_types) > 1
    ):
        return QueryType.CROSS_SOURCE
    if filters.last_n_quarters or filters.time_window or re.search(r"\b(changed|change|trend|over time)\b", lower):
        return QueryType.TEMPORAL
    if filters.speaker_names or filters.speaker_roles:
        return QueryType.SPEAKER_SPECIFIC
    return QueryType.SINGLE_DOC_LOOKUP


def _extract_tickers(question: str, *, default_ticker: str | None) -> list[str]:
    found: list[str] = []
    upper_question = question.upper()
    for ticker in re.findall(r"\b[A-Z]{2,5}\b", question):
        if ticker in KNOWN_TICKERS and ticker not in found:
            found.append(ticker)
    for company, ticker in COMPANY_TICKERS.items():
        if re.search(rf"\b{re.escape(company)}\b", upper_question) and ticker not in found:
            found.append(ticker)
    if not found and default_ticker:
        found.append(default_ticker.upper())
    return found


def _extract_company_names(question: str) -> list[str]:
    upper_question = question.upper()
    return [
        company.title()
        for company in COMPANY_TICKERS
        if re.search(rf"\b{re.escape(company)}\b", upper_question)
    ]


def _extract_form_types(question: str) -> list[str]:
    upper = question.upper()
    forms: list[str] = []
    for form in ("10-K", "10-Q", "8-K", "EX-99"):
        if form in upper and form not in forms:
            forms.append(form)
    lower = question.lower()
    if _asks_for_filing_risks(lower) and not any(form in forms for form in ("10-K", "10-Q")):
        forms.extend(["10-K", "10-Q"])
    if "cfo commentary" in lower and "EX-99" not in forms:
        forms.append("EX-99")
    if "press release" in lower and "EX-99" not in forms:
        forms.append("EX-99")
    return forms


def _extract_time_window(question: str) -> str:
    lower = question.lower()
    if re.search(r"\blast\s+year\b|\bover\s+the\s+last\s+year\b", lower):
        return "last_year"
    match = re.search(r"\blast\s+(\d+)\s+quarters?\b", lower)
    if match:
        return f"last_{match.group(1)}_quarters"
    return ""


def _extract_last_n_quarters(question: str) -> int | None:
    match = re.search(r"\blast\s+(\d+)\s+quarters?\b", question.lower())
    if match:
        return int(match.group(1))
    if re.search(r"\blast\s+year\b|\bover\s+the\s+last\s+year\b", question.lower()):
        return 4
    return None


def _extract_fiscal_periods(question: str) -> list[str]:
    periods = re.findall(r"\bFY\s?\d{4}\s?Q[1-4]\b|\bQ[1-4]\s?FY\s?\d{2,4}\b", question, flags=re.I)
    return [re.sub(r"\s+", " ", period.upper()) for period in periods]


def _extract_item_numbers(question: str) -> list[str]:
    items = re.findall(r"\bitem\s+(\d(?:[A-C])?|\d\.\d{2})\b", question, flags=re.I)
    normalized = [item.upper() for item in items]
    if _asks_for_filing_risks(question.lower()) and "1A" not in normalized:
        normalized.append("1A")
    return normalized


def _extract_document_roles(question: str) -> list[str]:
    lower = question.lower()
    roles: list[str] = []
    if any(term in lower for term in ("ex-99", "exhibit", "cfo commentary", "prepared remarks", "press release")):
        roles.append("exhibit")
    if any(term in lower for term in ("10-k", "10-q", "8-k", "filing", "risk factors", "item 1a")) or _asks_for_filing_risks(lower):
        roles.append("primary")
    return roles


def _extract_exhibit_types(question: str) -> list[str]:
    lower = question.lower()
    types: list[str] = []
    markers = (
        ("CFO_COMMENTARY", ("cfo commentary", "cfo")),
        ("PREPARED_REMARKS", ("prepared remarks", "prepared remark")),
        ("PRESS_RELEASE", ("press release", "earnings release")),
        ("PRESENTATION", ("slides", "presentation")),
        ("EX-99", ("ex-99", "exhibit 99")),
    )
    for value, words in markers:
        if any(word in lower for word in words) and value not in types:
            types.append(value)
    return types


def _extract_speaker_names(question: str) -> list[str]:
    names: list[str] = []
    for name in ("Jensen Huang", "Colette Kress"):
        if name.lower() in question.lower():
            names.append(name)
    return names


def _extract_speaker_roles(question: str) -> list[str]:
    lower = question.lower()
    roles: list[str] = []
    if "cfo" in lower or "chief financial officer" in lower:
        roles.append("CFO")
    if "ceo" in lower or "chief executive officer" in lower:
        roles.append("CEO")
    return roles


def _asks_for_filing_risks(lower_question: str) -> bool:
    if "cfo" in lower_question or "press release" in lower_question:
        return False
    return "item 1a" in lower_question or "risk factor" in lower_question or re.search(r"\brisks?\b", lower_question) is not None
