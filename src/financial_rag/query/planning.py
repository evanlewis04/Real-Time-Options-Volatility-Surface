"""Convert routed questions into deterministic retrieval subqueries."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.financial_rag.query.router import QueryFilters, RoutedQuery
from src.financial_rag.retrieval import RetrievalFilters
from src.financial_rag.scope import QueryType


@dataclass(frozen=True)
class RetrievalSubquery:
    subquery_id: str
    query: str
    filters: RetrievalFilters
    trace: dict[str, str | int | list[str]] = field(default_factory=dict)


def plan_retrieval(routed: RoutedQuery) -> list[RetrievalSubquery]:
    """Plan local dense retrieval calls for a routed query."""

    tickers = routed.filters.tickers or [None]
    if routed.query_type == QueryType.CROSS_SOURCE:
        return _build_cross_source_subqueries(routed, ticker=tickers[0])

    if routed.query_type == QueryType.CROSS_COMPANY:
        return [_build_subquery(routed, ticker=ticker, ordinal=index) for index, ticker in enumerate(tickers, 1)]

    if routed.query_type == QueryType.TEMPORAL and routed.filters.last_n_quarters:
        return [
            _build_subquery(
                routed,
                ticker=tickers[0],
                ordinal=index,
                extra_trace={"quarter_offset": index - 1, "time_window": routed.filters.time_window},
            )
            for index in range(1, routed.filters.last_n_quarters + 1)
        ]

    return [_build_subquery(routed, ticker=tickers[0], ordinal=1)]


def _build_cross_source_subqueries(routed: RoutedQuery, *, ticker: str | None) -> list[RetrievalSubquery]:
    filters = routed.filters
    subqueries: list[RetrievalSubquery] = []
    filing_forms = tuple(form for form in filters.form_types if form in {"10-K", "10-Q", "8-K"}) or ("10-K", "10-Q")
    if filing_forms or filters.item_numbers:
        subqueries.append(
            RetrievalSubquery(
                subquery_id="cross_source-01",
                query=_subquery_text(routed.question, filters, ticker=ticker, ordinal=1),
                filters=RetrievalFilters(
                    ticker=ticker,
                    form_type=filing_forms[0] if filing_forms else None,
                    form_types=filing_forms,
                    document_role="primary",
                    document_roles=("primary",),
                    item_number=_first(filters.item_numbers),
                    item_numbers=tuple(filters.item_numbers),
                ),
                trace={
                    "query_type": routed.query_type.value,
                    "ticker": ticker or "",
                    "source_slice": "filing",
                    "form_types": list(filing_forms),
                    "item_numbers": filters.item_numbers,
                },
            )
        )
    if filters.exhibit_types or "EX-99" in filters.form_types or "exhibit" in filters.document_roles:
        subqueries.append(
            RetrievalSubquery(
                subquery_id=f"cross_source-{len(subqueries) + 1:02d}",
                query=_subquery_text(routed.question, filters, ticker=ticker, ordinal=len(subqueries) + 1),
                filters=RetrievalFilters(
                    ticker=ticker,
                    form_type="EX-99",
                    form_types=("EX-99",),
                    document_role="exhibit",
                    document_roles=("exhibit",),
                    exhibit_type=_first(filters.exhibit_types),
                    exhibit_types=tuple(filters.exhibit_types),
                    speaker_role=_first(filters.speaker_roles),
                ),
                trace={
                    "query_type": routed.query_type.value,
                    "ticker": ticker or "",
                    "source_slice": "exhibit",
                    "form_types": ["EX-99"],
                    "exhibit_types": filters.exhibit_types,
                },
            )
        )
    return subqueries or [_build_subquery(routed, ticker=ticker, ordinal=1)]


def _build_subquery(
    routed: RoutedQuery,
    *,
    ticker: str | None,
    ordinal: int,
    extra_trace: dict[str, str | int | list[str]] | None = None,
) -> RetrievalSubquery:
    filters = routed.filters
    retrieval_filters = RetrievalFilters(
        ticker=ticker,
        form_type=_first(filters.form_types),
        form_types=tuple(filters.form_types),
        document_role=_first(filters.document_roles),
        document_roles=tuple(filters.document_roles),
        exhibit_type=_first(filters.exhibit_types),
        exhibit_types=tuple(filters.exhibit_types),
        item_number=_first(filters.item_numbers),
        item_numbers=tuple(filters.item_numbers),
        speaker_name=_first(filters.speaker_names),
        speaker_role=_first(filters.speaker_roles),
    )
    query = _subquery_text(routed.question, filters, ticker=ticker, ordinal=ordinal)
    trace: dict[str, str | int | list[str]] = {
        "query_type": routed.query_type.value,
        "ticker": ticker or "",
        "form_types": filters.form_types,
        "document_roles": filters.document_roles,
        "exhibit_types": filters.exhibit_types,
        "item_numbers": filters.item_numbers,
    }
    if extra_trace:
        trace.update(extra_trace)
    return RetrievalSubquery(
        subquery_id=f"{routed.query_type.value}-{ordinal:02d}",
        query=query,
        filters=retrieval_filters,
        trace=trace,
    )


def _subquery_text(
    question: str,
    filters: QueryFilters,
    *,
    ticker: str | None,
    ordinal: int,
) -> str:
    pieces = [question]
    if ticker:
        pieces.append(f"Ticker: {ticker}.")
    if filters.last_n_quarters:
        pieces.append(f"Temporal slice {ordinal} of {filters.last_n_quarters}.")
    if filters.item_numbers:
        pieces.append(f"SEC item {filters.item_numbers[0]}.")
    if filters.exhibit_types:
        pieces.append(filters.exhibit_types[0].replace("_", " ").title())
    return " ".join(pieces)


def _first(values: list[str]) -> str | None:
    return values[0] if values else None
