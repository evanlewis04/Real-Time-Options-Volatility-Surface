"""Voyage AI embedding provider with an idempotent local cache."""

from __future__ import annotations

from typing import Any, Protocol

from src.financial_rag.models import DocumentChunk, EmbeddingRecord
from src.financial_rag.storage import LocalRagStore


DEFAULT_VOYAGE_MODEL = "voyage-finance-2"
# A stalled or rate-limited embedding request must fail fast, not hang the whole
# ingest: an unbounded call was observed freezing a multi-hour run for 37+ min
# with no error. Bound every request and let the SDK auto-retry transient
# rate-limit / 5xx errors with backoff before giving up (a raised error is then
# recoverable — the batched fetch skips the name and the checkpoint resumes it).
DEFAULT_EMBED_TIMEOUT_SECONDS = 60.0
DEFAULT_EMBED_MAX_RETRIES = 3


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str], *, model: str, input_type: str) -> Any:
        """Return a provider-specific embedding response."""


class VoyageEmbeddingProvider:
    """Thin wrapper around the Voyage SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_VOYAGE_MODEL,
        client: EmbeddingClient | None = None,
        timeout: float = DEFAULT_EMBED_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_EMBED_MAX_RETRIES,
    ) -> None:
        self.model = model
        if client is not None:
            self.client = client
            return
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - depends on local optional install
            raise RuntimeError(
                "The voyageai package is required when VOYAGE_API_KEY is configured. "
                "Install project dependencies, then rerun the smoke script."
            ) from exc
        self.client = voyageai.Client(
            api_key=api_key, max_retries=max_retries, timeout=timeout
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(texts, model=self.model, input_type="document")
        embeddings = getattr(response, "embeddings", response)
        return [list(map(float, vector)) for vector in embeddings]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(texts, model=self.model, input_type="query")
        embeddings = getattr(response, "embeddings", response)
        return [list(map(float, vector)) for vector in embeddings]


class EmbeddingCache:
    """Store one embedding JSON file per chunk under `data/vector_cache/`."""

    def __init__(self, *, store: LocalRagStore, model: str = DEFAULT_VOYAGE_MODEL) -> None:
        self.store = store
        self.model = model

    def is_cached(self, chunk: DocumentChunk) -> bool:
        return self.store.embedding_path(chunk.chunk_id).exists()

    def write(self, chunk: DocumentChunk, embedding: list[float]) -> bool:
        record = EmbeddingRecord(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            model=self.model,
            embedding=embedding,
            metadata={
                "ticker": chunk.ticker,
                "cik": chunk.cik,
                "accession_number": chunk.accession_number,
                "form_type": chunk.form_type,
                "filing_date": chunk.filing_date,
                "source_url": chunk.source_url,
                "local_path": chunk.local_path,
                "document_role": chunk.document_role,
                "exhibit_type": chunk.exhibit_type,
                "section_path": chunk.section_path,
                "item_number": chunk.item_number,
                "speaker_name": chunk.speaker_name,
                "speaker_role": chunk.speaker_role,
                "chunk_start_offset": chunk.start_offset,
                "chunk_end_offset": chunk.end_offset,
            },
        )
        record_dict = record.to_dict()
        path = self.store.embedding_path(chunk.chunk_id)
        result = self.store.write_json_once(path, record_dict)
        # Only newly-cached chunks touch the manifest, and via an O(1) append: the
        # old wholesale `upsert_manifest` reread + rewrote this multi-hundred-MB
        # file on every single chunk (O(n^2) over a full-corpus run, ~20s/chunk at
        # scale — the true throughput wall, independent of the embedding API). The
        # per-chunk file write is idempotent, so a cache hit adds no duplicate row.
        if result.created:
            self.store.append_manifest(
                self.store.vector_cache_dir / "manifest.jsonl",
                record_dict,
            )
        return result.created


def embedding_setup_message() -> str:
    return (
        "VOYAGE_API_KEY is not configured. Add a real Voyage key to .env, then rerun "
        "`\\.\\venv\\Scripts\\python.exe scripts\\financial_rag_retrieval_repair.py --embed` "
        "to generate cached embeddings. OpenAI and Anthropic are not used for embeddings."
    )
