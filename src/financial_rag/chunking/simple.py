"""Deterministic character-window chunking for Phase 1 smoke tests."""

from __future__ import annotations

import hashlib
import re

from src.financial_rag.models import DocumentChunk, FilingMetadata
from src.financial_rag.parsing import PARSER_VERSION


CHUNKER_VERSION = "simple_char_window_v1"


def chunk_document(
    text: str,
    metadata: FilingMetadata,
    *,
    max_chars: int = 2200,
    overlap_chars: int = 200,
) -> list[DocumentChunk]:
    """Create stable chunks with offsets and source metadata."""

    normalized = text.strip()
    if not normalized:
        return []
    chunks: list[DocumentChunk] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        target_end = min(start + max_chars, text_length)
        end = _choose_boundary(normalized, start=start, target_end=target_end)
        chunk_text = normalized[start:end].strip()
        if chunk_text:
            chunks.append(_build_chunk(chunk_text, metadata, start, end, len(chunks)))
        if end >= text_length:
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def _choose_boundary(text: str, *, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)
    window = text[start:target_end]
    paragraph_break = window.rfind("\n\n")
    if paragraph_break > max(400, int(len(window) * 0.45)):
        return start + paragraph_break
    sentence_break = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if sentence_break > max(400, int(len(window) * 0.55)):
        return start + sentence_break + 1
    return target_end


def _build_chunk(
    chunk_text: str,
    metadata: FilingMetadata,
    start: int,
    end: int,
    ordinal: int,
) -> DocumentChunk:
    chunk_hash = hashlib.sha256(
        f"{metadata.document_id}|{start}|{end}|{chunk_text}".encode("utf-8")
    ).hexdigest()[:24]
    return DocumentChunk(
        chunk_id=f"{metadata.document_id}-chunk-{ordinal:04d}-{chunk_hash}",
        document_id=metadata.document_id,
        ticker=metadata.ticker,
        cik=metadata.cik,
        accession_number=metadata.accession_number,
        form_type=metadata.form_type,
        filing_date=metadata.filing_date,
        source_url=metadata.source_url,
        local_path=metadata.local_path,
        document_role=metadata.document_role,
        exhibit_type=metadata.exhibit_type,
        chunk_text=chunk_text,
        start_offset=start,
        end_offset=end,
        token_count=_approx_token_count(chunk_text),
        filed_at=metadata.filed_at,
        period_end=metadata.period_end,
        metadata={
            "parser_version": PARSER_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "report_date": metadata.report_date,
            "document_name": metadata.document_name,
            "description": metadata.description,
            "content_hash": metadata.content_hash,
        },
    )


def _approx_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
