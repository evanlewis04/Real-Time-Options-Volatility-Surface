"""Bounded parent-context hydration from local chunk records."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.financial_rag.retrieval import LocalChunkRecord, RetrievalResult


@dataclass(frozen=True)
class HydratedContext:
    chunk_id: str
    context_text: str
    context_chunk_ids: list[str] = field(default_factory=list)


def hydrate_parent_context(
    result: RetrievalResult,
    chunks: list[LocalChunkRecord],
    *,
    window: int = 1,
    max_chars: int = 5000,
) -> HydratedContext:
    """Hydrate nearby chunks from the same document as bounded parent context."""

    by_document = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("document_id") == result.metadata.get("document_id")
    ]
    by_document.sort(key=lambda chunk: int(chunk.metadata.get("start_offset", 0) or 0))
    index = next((idx for idx, chunk in enumerate(by_document) if chunk.chunk_id == result.chunk_id), None)
    if index is None:
        return HydratedContext(
            chunk_id=result.chunk_id,
            context_text=result.source_excerpt,
            context_chunk_ids=[result.chunk_id],
        )

    start = max(0, index - window)
    end = min(len(by_document), index + window + 1)
    selected = by_document[start:end]
    context_parts: list[str] = []
    context_ids: list[str] = []
    for chunk in selected:
        next_text = chunk.chunk_text.strip()
        candidate = "\n\n".join([*context_parts, next_text]) if context_parts else next_text
        if len(candidate) > max_chars and context_parts:
            break
        context_parts.append(next_text[:max_chars] if not context_parts else next_text)
        context_ids.append(chunk.chunk_id)
    return HydratedContext(
        chunk_id=result.chunk_id,
        context_text="\n\n".join(context_parts)[:max_chars],
        context_chunk_ids=context_ids,
    )
