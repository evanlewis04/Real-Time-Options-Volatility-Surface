"""Rerank stage over a retrieved candidate pool.

The default reranker is a deterministic, offline BM25-lite re-scorer that ranks
candidates against the original question (the first stage only sees per-subquery
text). An optional Voyage reranker is available behind a flag for online
comparison; it is never required, so offline evals stay reproducible.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from src.financial_rag.retrieval.local_dense import _content_terms, _metadata_relevance_bonus


@dataclass(frozen=True)
class RerankCandidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


class Reranker(Protocol):
    name: str

    def score(self, query: str, candidates: Sequence[RerankCandidate]) -> list[float]:
        """Return one relevance score per candidate; higher is more relevant."""


class LexicalRerankerV1:
    """Deterministic BM25-lite reranker scored against the original question."""

    name = "lexical_v1"

    def __init__(
        self,
        chunk_texts: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        metadata_weight: float = 1.5,
        phrase_weight: float = 1.0,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.metadata_weight = metadata_weight
        self.phrase_weight = phrase_weight
        doc_count = 0
        length_sum = 0
        document_frequency: Counter[str] = Counter()
        for text in chunk_texts:
            terms = _content_terms(text)
            doc_count += 1
            length_sum += len(terms)
            for term in set(terms):
                document_frequency[term] += 1
        self._doc_count = max(doc_count, 1)
        self._avg_doc_length = (length_sum / self._doc_count) if self._doc_count else 1.0
        self._document_frequency = document_frequency

    def score(self, query: str, candidates: Sequence[RerankCandidate]) -> list[float]:
        query_terms = _content_terms(query)
        query_phrases = _phrases(query)
        return [self._score_candidate(query_terms, query_phrases, query, candidate) for candidate in candidates]

    def _score_candidate(
        self,
        query_terms: list[str],
        query_phrases: list[str],
        query: str,
        candidate: RerankCandidate,
    ) -> float:
        text_lower = candidate.text.lower()
        doc_terms = _content_terms(candidate.text)
        if not doc_terms:
            return 0.0
        term_frequency = Counter(doc_terms)
        doc_length = len(doc_terms)
        bm25 = 0.0
        for term in query_terms:
            term_count = term_frequency.get(term, 0)
            if term_count == 0:
                continue
            idf = self._idf(term)
            denominator = term_count + self.k1 * (1.0 - self.b + self.b * doc_length / self._avg_doc_length)
            bm25 += idf * (term_count * (self.k1 + 1.0)) / denominator
        phrase_bonus = self.phrase_weight * sum(1.0 for phrase in query_phrases if phrase in text_lower)
        metadata_bonus = self.metadata_weight * _metadata_relevance_bonus(query, candidate.metadata)
        return bm25 + phrase_bonus + metadata_bonus

    def _idf(self, term: str) -> float:
        df = self._document_frequency.get(term, 0)
        return math.log(1.0 + (self._doc_count - df + 0.5) / (df + 0.5))


class VoyageReranker:
    """Optional online reranker using the Voyage rerank API (lazy import)."""

    name = "voyage"

    def __init__(self, *, model: str = "rerank-2", api_key: str | None = None) -> None:
        from src.financial_rag.settings import configured_secret, load_environment

        load_environment()
        resolved_key = api_key or configured_secret("VOYAGE_API_KEY")
        if not resolved_key:
            raise RuntimeError("VOYAGE_API_KEY is required for the Voyage reranker.")
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The voyageai package is not installed.") from exc
        self._client = voyageai.Client(api_key=resolved_key)
        self._model = model

    def score(self, query: str, candidates: Sequence[RerankCandidate]) -> list[float]:  # pragma: no cover - network
        if not candidates:
            return []
        documents = [candidate.text for candidate in candidates]
        response = self._client.rerank(query=query, documents=documents, model=self._model)
        scores = [0.0] * len(candidates)
        for item in response.results:
            scores[item.index] = float(item.relevance_score)
        return scores


def build_reranker(name: str, *, chunk_texts: Sequence[str]) -> Reranker | None:
    """Construct a reranker by name. 'none' disables reranking."""

    normalized = (name or "none").strip().lower()
    if normalized in {"none", "off", ""}:
        return None
    if normalized in {"lexical", "lexical_v1", "bm25"}:
        return LexicalRerankerV1(chunk_texts)
    if normalized == "voyage":
        return VoyageReranker()
    raise ValueError(f"Unknown reranker '{name}'. Use 'none', 'lexical', or 'voyage'.")


def _phrases(query: str) -> list[str]:
    lower = query.lower()
    words = lower.split()
    return [f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1) if len(words[index]) > 2]
