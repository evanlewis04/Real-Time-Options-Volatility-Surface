"""Thin integration between filings RAG evidence and market context."""

from .market_evidence import (
    FILING_EVIDENCE_LABEL,
    MARKET_CONTEXT_LABEL,
    MarketEvidenceBrief,
    build_brief_from_service,
    build_market_evidence_brief,
    market_provider_from_metrics,
    volatility_market_provider,
)

__all__ = [
    "FILING_EVIDENCE_LABEL",
    "MARKET_CONTEXT_LABEL",
    "MarketEvidenceBrief",
    "build_brief_from_service",
    "build_market_evidence_brief",
    "market_provider_from_metrics",
    "volatility_market_provider",
]
