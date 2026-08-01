"""Phase 1 Stage 3: corpus snapshots make drift unable to move a *pinned* run.

The headline guarantee (``test_pinned_read_reproduces_after_new_docs_land``):
snapshot the corpus, then land a new document, and a run pinned to the snapshot
returns the exact same ranked results — while an unpinned run picks up the new
document. That is the drift-forward defense in one assertion, exercised on a
synthetic on-disk corpus (the deterministic vehicle, mirroring Stage 2), since
the real corpus composition cannot be controlled.

Honest scope, asserted directly: a snapshot pins state from the moment it is
taken. A document mutated after the snapshot is *excluded and reported as drift*,
never silently served with its newer content — snapshots detect the loss they
cannot resurrect.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.financial_rag.corpus_snapshot import (
    build_corpus_snapshot,
    list_corpus_snapshots,
    read_corpus_snapshot,
    restrict_chunks_to_snapshot,
    snapshot_drift,
    write_corpus_snapshot,
)
from src.financial_rag.retrieval import (
    LocalChunkRecord,
    LocalDenseRetriever,
    load_local_retrieval_corpus,
)
from src.financial_rag.storage import LocalRagStore

_FIXED_STAMP = "2026-08-01T00:00:00+00:00"


def _chunk(document_id: str, chunk_id: str, text: str, *, content_hash: str) -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "accession_number": f"acc-{document_id}",
            "filed_at": "2026-01-15T12:00:00+00:00",
            "content_hash": content_hash,
        },
    )


def _write_doc(store: LocalRagStore, document_id: str, *, content_hash: str, chunks: list[tuple[str, str]]) -> None:
    """Write a document's chunk JSONL file in the on-disk corpus layout."""

    rows = [
        {
            "chunk_id": chunk_id,
            "chunk_text": text,
            "document_id": document_id,
            "accession_number": f"acc-{document_id}",
            "filed_at": "2026-01-15T12:00:00+00:00",
            "metadata": {"content_hash": content_hash},
        }
        for chunk_id, text in chunks
    ]
    path = store.chunks_path(document_id)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _ranked(chunks: list[LocalChunkRecord], query: str, *, top_k: int = 5) -> list[tuple[str, float]]:
    """Deterministic offline ranking: no embeddings on disk, so lexical drives it."""

    retriever = LocalDenseRetriever(chunks=chunks, embeddings={}, query_embedder=None)
    results = retriever.search(query=query, query_vector=[0.0], top_k=top_k)
    return [(result.chunk_id, round(result.score, 6)) for result in results]


# --- snapshot identity + persistence --------------------------------------


def test_snapshot_id_is_deterministic_over_composition_not_time() -> None:
    chunks = [
        _chunk("doc-a", "a0", "export controls data center revenue", content_hash="hash-a"),
        _chunk("doc-b", "b0", "gaming seasonal demand", content_hash="hash-b"),
    ]
    first = build_corpus_snapshot(chunks, created_at="2026-01-01T00:00:00+00:00")
    second = build_corpus_snapshot(chunks, created_at="2026-12-31T23:59:59+00:00")
    assert first.snapshot_id == second.snapshot_id
    assert first.document_count == 2
    assert first.chunk_count == 2

    changed = build_corpus_snapshot(
        chunks + [_chunk("doc-c", "c0", "new filing", content_hash="hash-c")],
        created_at=_FIXED_STAMP,
    )
    assert changed.snapshot_id != first.snapshot_id


def test_snapshot_roundtrips_through_manifest_idempotently(tmp_path: Path) -> None:
    store = LocalRagStore(root=tmp_path)
    chunks = [_chunk("doc-a", "a0", "export controls", content_hash="hash-a")]
    snapshot = build_corpus_snapshot(chunks, created_at=_FIXED_STAMP)

    assert write_corpus_snapshot(store, snapshot) is True
    assert write_corpus_snapshot(store, snapshot) is False  # idempotent second write

    loaded = read_corpus_snapshot(store, snapshot.snapshot_id)
    assert loaded is not None
    assert loaded.snapshot_id == snapshot.snapshot_id
    assert loaded.document_ids() == {"doc-a"}
    assert [s.snapshot_id for s in list_corpus_snapshots(store)] == [snapshot.snapshot_id]
    assert read_corpus_snapshot(store, "sha256:does-not-exist") is None


# --- restrict + drift semantics -------------------------------------------


def test_restrict_excludes_documents_added_after_snapshot() -> None:
    snapshotted = [_chunk("doc-a", "a0", "text", content_hash="hash-a")]
    snapshot = build_corpus_snapshot(snapshotted, created_at=_FIXED_STAMP)

    grown = snapshotted + [_chunk("doc-b", "b0", "new", content_hash="hash-b")]
    kept = restrict_chunks_to_snapshot(grown, snapshot)
    assert [chunk.chunk_id for chunk in kept] == ["a0"]


def test_restrict_excludes_mutated_document() -> None:
    original = [_chunk("doc-a", "a0", "original text", content_hash="hash-a")]
    snapshot = build_corpus_snapshot(original, created_at=_FIXED_STAMP)

    mutated = [_chunk("doc-a", "a0-new", "rewritten text", content_hash="hash-a-v2")]
    kept = restrict_chunks_to_snapshot(mutated, snapshot)
    assert kept == []  # newer content is never silently served under the pin


def test_snapshot_drift_reports_added_removed_mutated() -> None:
    snapshot = build_corpus_snapshot(
        [
            _chunk("doc-a", "a0", "t", content_hash="hash-a"),
            _chunk("doc-b", "b0", "t", content_hash="hash-b"),
        ],
        created_at=_FIXED_STAMP,
    )
    current = [
        _chunk("doc-a", "a0", "t", content_hash="hash-a"),  # unchanged
        _chunk("doc-b", "b0-new", "t", content_hash="hash-b-v2"),  # mutated
        _chunk("doc-c", "c0", "t", content_hash="hash-c"),  # added
    ]
    drift = snapshot_drift(current, snapshot)
    assert drift == {"added": ["doc-c"], "removed": [], "mutated": ["doc-b"]}


# --- headline: pinned reproduction over the real load seam ------------------


def test_pinned_read_reproduces_after_new_docs_land(tmp_path: Path) -> None:
    store = LocalRagStore(root=tmp_path)
    query = "export controls data center revenue"

    _write_doc(store, "doc-a", content_hash="hash-a", chunks=[("a0", "export controls data center revenue guidance")])
    _write_doc(store, "doc-b", content_hash="hash-b", chunks=[("b0", "gaming segment seasonal demand commentary")])

    baseline_chunks, _ = load_local_retrieval_corpus(root=tmp_path)
    snapshot = build_corpus_snapshot(baseline_chunks, created_at=_FIXED_STAMP)
    assert write_corpus_snapshot(store, snapshot) is True
    baseline_ranking = _ranked(baseline_chunks, query)

    # A new filing lands (the exact drift Stage 3 defends against). It would rank
    # for the query, so an unpinned run must move.
    _write_doc(
        store,
        "doc-c",
        content_hash="hash-c",
        chunks=[("c0", "export controls data center revenue and outlook detail")],
    )

    pinned = read_corpus_snapshot(store, snapshot.snapshot_id)
    assert pinned is not None
    pinned_chunks, _ = load_local_retrieval_corpus(root=tmp_path, snapshot=pinned)
    unpinned_chunks, _ = load_local_retrieval_corpus(root=tmp_path)

    # Pinned run is numerically identical to the pre-drift baseline; unpinned moved.
    assert _ranked(pinned_chunks, query) == baseline_ranking
    assert {chunk.chunk_id for chunk in pinned_chunks} == {"a0", "b0"}

    unpinned_ids = {chunk.chunk_id for chunk in unpinned_chunks}
    assert unpinned_ids == {"a0", "b0", "c0"}
    assert "c0" in {chunk_id for chunk_id, _ in _ranked(unpinned_chunks, query)}
    assert _ranked(unpinned_chunks, query) != baseline_ranking

    # And the snapshot names the drift honestly.
    assert snapshot_drift(unpinned_chunks, pinned) == {"added": ["doc-c"], "removed": [], "mutated": []}


def test_snapshot_pin_is_a_noop_when_none(tmp_path: Path) -> None:
    store = LocalRagStore(root=tmp_path)
    _write_doc(store, "doc-a", content_hash="hash-a", chunks=[("a0", "text one")])
    _write_doc(store, "doc-b", content_hash="hash-b", chunks=[("b0", "text two")])

    unpinned, _ = load_local_retrieval_corpus(root=tmp_path)
    explicit_none, _ = load_local_retrieval_corpus(root=tmp_path, snapshot=None)
    assert {chunk.chunk_id for chunk in unpinned} == {"a0", "b0"}
    assert [chunk.chunk_id for chunk in explicit_none] == [chunk.chunk_id for chunk in unpinned]
