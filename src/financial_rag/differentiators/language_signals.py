"""Transparent keyword language signals for filings chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.financial_rag.retrieval import LocalChunkRecord


UNCERTAINTY_TERMS = ("may", "could", "might", "uncertain", "uncertainty", "depend", "subject to")
RISK_TERMS = ("risk", "adverse", "material", "decline", "constraint", "competition", "regulation")
POSITIVE_TERMS = ("increase", "growth", "improve", "strong", "benefit", "opportunity", "demand")
NEGATIVE_TERMS = ("decrease", "decline", "weak", "loss", "shortage", "delay", "adverse")


@dataclass(frozen=True)
class LanguageSignalSummary:
    scope: str
    ticker: str
    document_id: str
    item_number: str
    chunk_count: int
    token_count: int
    uncertainty_hits: int
    risk_hits: int
    positive_hits: int
    negative_hits: int
    sentiment_score: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def score_language_signals(chunks: list[LocalChunkRecord]) -> list[LanguageSignalSummary]:
    """Return per-chunk transparent keyword scores."""

    summaries: list[LanguageSignalSummary] = []
    for chunk in chunks:
        summaries.append(_summary([chunk], scope="chunk"))
    return summaries


def summarize_language_signals(
    chunks: list[LocalChunkRecord],
    *,
    group_by: str = "document_id",
) -> list[LanguageSignalSummary]:
    """Summarize language signals by document or item metadata."""

    grouped: dict[str, list[LocalChunkRecord]] = {}
    for chunk in chunks:
        key = str(chunk.metadata.get(group_by, "")).strip() or "unknown"
        grouped.setdefault(key, []).append(chunk)
    return [_summary(group, scope=f"{group_by}:{key}") for key, group in sorted(grouped.items())]


def _summary(chunks: list[LocalChunkRecord], *, scope: str) -> LanguageSignalSummary:
    text = " ".join(chunk.chunk_text for chunk in chunks)
    positive = _count_terms(text, POSITIVE_TERMS)
    negative = _count_terms(text, NEGATIVE_TERMS)
    first = chunks[0] if chunks else LocalChunkRecord("", "", {})
    return LanguageSignalSummary(
        scope=scope,
        ticker=str(first.metadata.get("ticker", "")),
        document_id=str(first.metadata.get("document_id", "")),
        item_number=str(first.metadata.get("item_number", "")),
        chunk_count=len(chunks),
        token_count=len(re.findall(r"\S+", text)),
        uncertainty_hits=_count_terms(text, UNCERTAINTY_TERMS),
        risk_hits=_count_terms(text, RISK_TERMS),
        positive_hits=positive,
        negative_hits=negative,
        sentiment_score=positive - negative,
    )


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    lowered = text.lower()
    total = 0
    for term in terms:
        total += len(re.findall(rf"\b{re.escape(term)}\b", lowered))
    return total
