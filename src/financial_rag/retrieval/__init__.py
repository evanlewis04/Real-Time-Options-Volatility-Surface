"""Local retrieval primitives for filings RAG."""

from .local_dense import (
    LocalDenseRetriever,
    LocalChunkRecord,
    RetrievalFilters,
    RetrievalResult,
    cosine_similarity,
    lexical_relevance_score,
    load_local_retrieval_corpus,
)

__all__ = [
    "LocalDenseRetriever",
    "LocalChunkRecord",
    "RetrievalFilters",
    "RetrievalResult",
    "cosine_similarity",
    "lexical_relevance_score",
    "load_local_retrieval_corpus",
]
