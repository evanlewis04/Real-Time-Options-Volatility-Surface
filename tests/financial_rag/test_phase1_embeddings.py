import importlib.util
import sys
import types
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


def test_embedding_cache_appends_manifest_without_rewriting(tmp_path: Path) -> None:
    """The embedding manifest must be appended, not rewritten wholesale, per chunk.

    Regression: writing each embedding used to reread + re-serialize + rewrite the
    entire vector-cache manifest (O(n^2) over a run, ~20s/chunk once the file grew
    into the hundreds of MB — the real throughput wall behind the API rate cap).
    New chunks now append one row; a cache hit adds none; pre-existing content is
    left byte-for-byte intact (proving no wholesale rewrite).
    """
    store = LocalRagStore(root=tmp_path)
    cache = EmbeddingCache(store=store, model="fake-model")
    manifest_path = store.vector_cache_dir / "manifest.jsonl"
    # A sentinel row that must survive untouched if we only ever append.
    sentinel = '{"chunk_id": "pre-existing", "embedding": [9.9]}\n'
    manifest_path.write_text(sentinel, encoding="utf-8")

    chunk = _make_chunk(1)
    assert cache.write(chunk, [0.1, 0.2]) is True  # new -> one appended row
    assert cache.write(chunk, [0.3, 0.4]) is False  # cache hit -> no new row

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == sentinel.rstrip("\n")  # original content preserved verbatim
    assert len(lines) == 2  # exactly one row appended, no duplicate on the re-write
    assert lines[1].count('"chunk_id": "chunk-1"') == 1


# --------------------------------------------------------------------------- #
# Embedding throttle: opt-in pacing keeps a capped free key under its rate cap.
# --------------------------------------------------------------------------- #


def _make_chunk(idx: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{idx}",
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
        chunk_text=f"text-{idx}",
        start_offset=0,
        end_offset=5,
        token_count=1,
        metadata={"parser_version": "test"},
    )


def _load_ingest_engine():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "financial_rag_ingest", root / "scripts" / "financial_rag_ingest.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeProvider:
    model = "fake-model"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


def _stub_embed_dependencies(engine, monkeypatch, sleeps: list[float]) -> None:
    monkeypatch.setattr(engine, "configured_secret", lambda name: "test-key")
    monkeypatch.setattr(engine, "VoyageEmbeddingProvider", lambda **kwargs: _FakeProvider())
    # Freeze the clock so each post-first request must wait the full interval, and
    # capture sleeps instead of actually blocking the test.
    monkeypatch.setattr(
        engine,
        "time",
        types.SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda s: sleeps.append(s)),
    )


def test_embed_chunks_throttles_between_requests_when_interval_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A positive min_request_interval spaces requests (the free-key rate-cap path)."""
    engine = _load_ingest_engine()
    store = LocalRagStore(root=tmp_path)
    sleeps: list[float] = []
    _stub_embed_dependencies(engine, monkeypatch, sleeps)

    counts = engine.SmokeCounts()
    engine.embed_chunks(
        store=store,
        chunks=[_make_chunk(i) for i in range(3)],
        counts=counts,
        batch_size=1,
        min_request_interval=20.0,
    )

    assert counts.embedded == 3
    # 3 one-chunk requests -> 2 inter-request waits, each the full interval.
    assert sleeps == [20.0, 20.0]


def test_embed_chunks_does_not_throttle_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default interval 0.0 preserves the original unthrottled (paid-key) behavior."""
    engine = _load_ingest_engine()
    store = LocalRagStore(root=tmp_path)
    sleeps: list[float] = []
    _stub_embed_dependencies(engine, monkeypatch, sleeps)

    counts = engine.SmokeCounts()
    engine.embed_chunks(
        store=store,
        chunks=[_make_chunk(i) for i in range(3)],
        counts=counts,
        batch_size=1,
    )

    assert counts.embedded == 3
    assert sleeps == []
