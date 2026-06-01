"""Combine cited filing evidence with options-market context, kept separate.

This is a thin, cache-only prototype. Filing evidence is management disclosure
with validated citations; market context is options-market-implied data with its
own provenance. The two are never merged into a single claim: data-source labels
stay explicit so a reader can tell what management disclosed from what the
options market is implying. The volatility engine plugs in through an injectable
provider, so this package does not depend on the dashboard or quant stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.financial_rag.api import QueryRequest
from src.financial_rag.differentiators import MarketContext, get_market_context


FILING_EVIDENCE_LABEL = "sec_filings_disclosure"
MARKET_CONTEXT_LABEL = "options_market_implied"

# Market metric keys that are not part of the metrics payload.
_NON_METRIC_KEYS = frozenset({"source_mode", "mode", "message"})


@dataclass(frozen=True)
class MarketEvidenceBrief:
    """A combined brief separating cited disclosure from market-implied context."""

    question: str
    ticker: str
    filing_evidence: dict[str, Any]
    market_context: dict[str, Any]
    data_sources: list[dict[str, str]]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ticker": self.ticker,
            "filing_evidence": self.filing_evidence,
            "market_context": self.market_context,
            "data_sources": self.data_sources,
            "notes": list(self.notes),
        }


def build_market_evidence_brief(
    query_payload: dict[str, Any],
    market_context: MarketContext,
    *,
    question: str,
    ticker: str,
) -> MarketEvidenceBrief:
    """Merge a RAG query payload and a market context into a labeled brief."""

    results = list(query_payload.get("results", []))
    citations = query_payload.get("citations", {})
    filing_evidence = {
        "source": FILING_EVIDENCE_LABEL,
        "description": "Management disclosure from SEC filings, with validated citations.",
        "result_count": len(results),
        "evidence": [
            {
                "label": str(result.get("citation_label", "")),
                "ticker": str(result.get("metadata", {}).get("ticker", "")),
                "form_type": str(result.get("metadata", {}).get("form_type", "")),
                "filing_date": str(result.get("metadata", {}).get("filing_date", "")),
                "source_url": str(result.get("source_url", "")),
                "excerpt": str(result.get("source_excerpt", "")),
            }
            for result in results
        ],
        "accepted_citations": list(citations.get("accepted", [])),
        "rejected_citations": list(citations.get("rejected", [])),
    }

    market_payload = market_context.to_dict()
    market_block = {
        "source": MARKET_CONTEXT_LABEL,
        "description": "Options-market-implied context (e.g. expected move, IV rank, skew).",
        "status": str(market_payload.get("status", "")),
        "source_mode": str(market_payload.get("source_mode", "")),
        "message": str(market_payload.get("message", "")),
        "metrics": dict(market_payload.get("metrics", {})),
    }

    notes = [
        "Filing evidence is management disclosure; market context is market-implied "
        "and is not filing evidence.",
    ]
    if market_block["status"] != "ok":
        notes.append(
            "Market context is unavailable or labeled; do not infer market reaction "
            "from filings alone."
        )

    data_sources = [
        {"label": FILING_EVIDENCE_LABEL, "kind": "disclosure", "provenance": "SEC EDGAR filings, cited"},
        {
            "label": MARKET_CONTEXT_LABEL,
            "kind": "market_data",
            "provenance": market_block["source_mode"] or market_block["status"] or "unavailable",
        },
    ]

    return MarketEvidenceBrief(
        question=question,
        ticker=ticker.upper(),
        filing_evidence=filing_evidence,
        market_context=market_block,
        data_sources=data_sources,
        notes=notes,
    )


def build_brief_from_service(
    service: Any,
    *,
    question: str,
    ticker: str,
    top_k: int = 5,
    per_subquery_k: int = 5,
    market_provider: Callable[[str], dict[str, Any]] | None = None,
) -> MarketEvidenceBrief:
    """Cache-only convenience: run local RAG retrieval and attach market context.

    ``market_provider`` is injectable. When omitted, the market context is labeled
    unavailable and the path stays fully offline; pass ``volatility_market_provider``
    (or any provider) to attach live or fallback market metrics.
    """

    query_payload = service.query(
        QueryRequest(question=question, ticker=ticker, top_k=top_k, per_subquery_k=per_subquery_k)
    )
    market_context = get_market_context(ticker, provider=market_provider)
    return build_market_evidence_brief(query_payload, market_context, question=question, ticker=ticker)


def market_provider_from_metrics(snapshot: dict[str, Any]) -> Callable[[str], dict[str, Any]]:
    """Build a market-context provider from a precomputed market snapshot.

    This is the seam where the existing volatility engine plugs in: a caller
    produces a market snapshot (carrying ``source_mode`` plus market-implied
    metrics such as expected move, IV rank, and skew) and this adapter exposes it
    as a provider without coupling the RAG package to the dashboard.
    """

    metrics = {key: value for key, value in snapshot.items() if key not in _NON_METRIC_KEYS}

    def _provider(_ticker: str) -> dict[str, Any]:
        return {
            "source_mode": str(snapshot.get("source_mode", snapshot.get("mode", ""))),
            "message": str(snapshot.get("message", "Market context from volatility engine.")),
            "metrics": metrics,
        }

    return _provider


def volatility_market_provider(ticker: str) -> dict[str, Any]:
    """Provider backed by the existing volatility engine (lazy, optional).

    Imports the dashboard connector lazily so the RAG package never hard-depends
    on the dashboard/quant stack. Any failure (missing stack, no market data)
    propagates to ``get_market_context``, which labels the context unavailable.
    """

    from dashboard_connector import DashboardConnector

    connector = DashboardConnector()
    data = connector.get_current_data(ticker.upper())
    metric_keys = (
        "price",
        "iv_30d",
        "iv_rank",
        "expected_move",
        "expected_move_pct",
        "front_expected_move_pct",
        "skew",
        "term_structure",
        "timestamp",
    )
    return {
        "source_mode": str(data.get("data_mode", data.get("source_mode", ""))),
        "message": "Market context from the volatility engine.",
        "metrics": {key: data[key] for key in metric_keys if key in data},
    }
