from pathlib import Path

import pytest

from src.financial_rag.embeddings import EmbeddingCache, VoyageEmbeddingProvider
from src.financial_rag.models import DocumentChunk
from src.financial_rag.storage import LocalRagStore


class FakeVoyageResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class FakeVoyageClient:
    def embed(self, texts: list[str], *, model: str, input_type: str) -> FakeVoyageResponse:
        assert model == "fake-model"
        assert input_type == "document"
        return FakeVoyageResponse([[float(len(text)), 1.0] for text in texts])


def test_voyage_provider_accepts_fake_client_without_network() -> None:
    provider = VoyageEmbeddingProvider(
        api_key="test-key",
        model="fake-model",
        client=FakeVoyageClient(),
    )

    assert provider.embed_texts(["abc", "hello"]) == [[3.0, 1.0], [5.0, 1.0]]


def test_real_client_is_built_with_finite_timeout_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the live Voyage client must have a request timeout + retries.

    An unbounded client (timeout=None, max_retries=0) let a single stalled or
    rate-limited embedding request hang a multi-hour ingest indefinitely. The
    provider must pass a finite timeout and at least one retry so a wedged call
    fails fast (and, being retriable, either recovers or surfaces a real error).
    """
    voyageai = pytest.importorskip("voyageai")
    captured: dict[str, object] = {}

    class RecordingClient:
        def __init__(self, api_key=None, max_retries=0, timeout=None, base_url=None):
            captured.update(api_key=api_key, max_retries=max_retries, timeout=timeout)

    monkeypatch.setattr(voyageai, "Client", RecordingClient)

    VoyageEmbeddingProvider(api_key="pa-test-key")

    assert isinstance(captured["timeout"], (int, float)) and captured["timeout"] > 0
    assert isinstance(captured["max_retries"], int) and captured["max_retries"] >= 1


def test_embedding_cache_writes_once_with_source_metadata(tmp_path: Path) -> None:
    store = LocalRagStore(root=tmp_path)
    cache = EmbeddingCache(store=store, model="fake-model")
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        ticker="NVDA",
        cik="0001045810",
        accession_number="0001045810-26-000051",
        form_type="10-Q",
        filing_date="2026-05-20",
        source_url="https://www.sec.gov/Archives/doc.htm",
        local_path="data/filings/raw/NVDA/doc.htm",
        document_role="primary",
        exhibit_type="",
        chunk_text="hello",
        start_offset=0,
        end_offset=5,
        token_count=1,
        metadata={"parser_version": "test"},
    )

    first = cache.write(chunk, [0.1, 0.2])
    second = cache.write(chunk, [0.3, 0.4])
    cached_text = store.embedding_path(chunk.chunk_id).read_text(encoding="utf-8")

    assert first is True
    assert second is False
    assert cache.is_cached(chunk) is True
    assert '"model": "fake-model"' in cached_text
    assert '"source_url": "https://www.sec.gov/Archives/doc.htm"' in cached_text
    assert '"embedding": [' in cached_text
