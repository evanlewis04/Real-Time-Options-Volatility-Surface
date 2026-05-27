"""Deterministic chunking strategies for filings RAG."""

from .strategies import CHUNKER_VERSION, chunk_sec_document

chunk_document = chunk_sec_document

__all__ = ["CHUNKER_VERSION", "chunk_document", "chunk_sec_document"]
