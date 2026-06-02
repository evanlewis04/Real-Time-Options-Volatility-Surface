"""Thin integration between filings RAG evidence and market context."""

from .brief import UnifiedBrief, assemble_unified_brief, build_unified_brief
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
    "UnifiedBrief",
    "assemble_unified_brief",
    "build_brief_from_service",
    "build_market_evidence_brief",
    "build_unified_brief",
    "market_provider_from_metrics",
    "volatility_market_provider",
]
