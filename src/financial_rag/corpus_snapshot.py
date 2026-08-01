"""Corpus snapshots: record and replay corpus composition (Phase 1 Stage 3).

A *snapshot* records which documents — and at what content — composed the corpus
at the moment it was taken, so an eval restricted to snapshot ``S`` reproduces
even after new filings land. This makes drift structurally unable to move a
*pinned* run **going forward**.

Honest scope (stated, not pretended): a snapshot pins state only from the moment
it is taken. It cannot resurrect content that already rolled forward — chunks
overwritten before any snapshot existed are gone, and that irreversible loss is
exactly the drift this defends against. A snapshot detects that a document has
since drifted (content-hash mismatch); it does not reconstruct the old content.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable

from src.financial_rag.storage import LocalRagStore
from src.financial_rag.storage.local_store import read_jsonl

if TYPE_CHECKING:  # avoid a runtime import cycle (retrieval imports this module).
    from src.financial_rag.retrieval import LocalChunkRecord


@dataclass(frozen=True)
class DocumentComposition:
    """One document's contribution to a corpus snapshot."""

    document_id: str
    accession: str
    filed_at: str
    content_hash: str
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "accession": self.accession,
            "filed_at": self.filed_at,
            "content_hash": self.content_hash,
            "chunk_count": self.chunk_count,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DocumentComposition":
        return cls(
            document_id=str(row.get("document_id", "")),
            accession=str(row.get("accession", "")),
            filed_at=str(row.get("filed_at", "")),
            content_hash=str(row.get("content_hash", "")),
            chunk_count=int(row.get("chunk_count", 0)),
        )


@dataclass(frozen=True)
class CorpusSnapshot:
    """Recorded composition of the corpus at a moment in time."""

    snapshot_id: str
    created_at: str
    document_count: int
    chunk_count: int
    documents: tuple[DocumentComposition, ...]

    def document_ids(self) -> set[str]:
        return {doc.document_id for doc in self.documents}

    def content_hash_by_document(self) -> dict[str, str]:
        return {doc.document_id: doc.content_hash for doc in self.documents}

    def to_row(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "documents": [doc.to_dict() for doc in self.documents],
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CorpusSnapshot":
        documents = tuple(
            DocumentComposition.from_dict(entry)
            for entry in row.get("documents", [])
            if isinstance(entry, dict)
        )
        return cls(
            snapshot_id=str(row.get("snapshot_id", "")),
            created_at=str(row.get("created_at", "")),
            document_count=int(row.get("document_count", len(documents))),
            chunk_count=int(row.get("chunk_count", 0)),
            documents=documents,
        )


def _composition_fingerprint(documents: Iterable[DocumentComposition]) -> str:
    """Deterministic id over the sorted (document, content, size) composition.

    Changes if any document is added, removed, mutated (content hash differs), or
    re-chunked (chunk count differs) — so the id identifies *which* corpus a run
    used, independent of when the snapshot was written.
    """

    payload = "\n".join(
        f"{doc.document_id}\t{doc.content_hash}\t{doc.chunk_count}"
        for doc in sorted(documents, key=lambda doc: doc.document_id)
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_corpus_snapshot(
    chunks: Iterable["LocalChunkRecord"],
    *,
    created_at: str | None = None,
) -> CorpusSnapshot:
    """Derive a snapshot from already-loaded chunk records.

    Groups chunks by ``document_id`` and records per document its accession,
    filed_at, content hash, and chunk count. ``created_at`` is injectable so
    tests and reruns stay deterministic; it never feeds the snapshot id.
    """

    grouped: dict[str, list["LocalChunkRecord"]] = {}
    for chunk in chunks:
        document_id = str(chunk.metadata.get("document_id", ""))
        if not document_id:
            continue
        grouped.setdefault(document_id, []).append(chunk)

    documents: list[DocumentComposition] = []
    for document_id in sorted(grouped):
        members = grouped[document_id]
        first = members[0].metadata
        documents.append(
            DocumentComposition(
                document_id=document_id,
                accession=str(first.get("accession_number", "")),
                filed_at=str(first.get("filed_at", "")),
                content_hash=str(first.get("content_hash", "")),
                chunk_count=len(members),
            )
        )

    total_chunks = sum(doc.chunk_count for doc in documents)
    stamp = created_at if created_at is not None else datetime.now(timezone.utc).isoformat()
    return CorpusSnapshot(
        snapshot_id=_composition_fingerprint(documents),
        created_at=stamp,
        document_count=len(documents),
        chunk_count=total_chunks,
        documents=tuple(documents),
    )


def write_corpus_snapshot(store: LocalRagStore, snapshot: CorpusSnapshot) -> bool:
    """Persist a snapshot to the JSONL snapshot manifest (idempotent).

    Returns True when the manifest content changed. Reuses ``upsert_manifest``
    keyed by ``snapshot_id`` — one row per snapshot, no new store invented.
    """

    return store.upsert_manifest(
        store.snapshot_manifest_path(),
        key="snapshot_id",
        record=snapshot.to_row(),
    )


def read_corpus_snapshot(store: LocalRagStore, snapshot_id: str) -> CorpusSnapshot | None:
    """Read one recorded snapshot by id, or None when it is not present."""

    for row in read_jsonl(store.snapshot_manifest_path()):
        if str(row.get("snapshot_id", "")) == snapshot_id:
            return CorpusSnapshot.from_row(row)
    return None


def list_corpus_snapshots(store: LocalRagStore) -> list[CorpusSnapshot]:
    """List all recorded snapshots (manifest order: sorted by snapshot id)."""

    return [CorpusSnapshot.from_row(row) for row in read_jsonl(store.snapshot_manifest_path())]


def restrict_chunks_to_snapshot(
    chunks: Iterable["LocalChunkRecord"],
    snapshot: CorpusSnapshot,
) -> list["LocalChunkRecord"]:
    """Keep only chunks whose document is in the snapshot at the recorded content.

    A chunk is kept when its ``document_id`` is in the snapshot **and** its
    ``content_hash`` matches what the snapshot recorded. Consequences:

    * Documents added after the snapshot (new ``document_id``) are excluded — so
      new filings landing cannot move a pinned run. This is the going-forward
      guarantee.
    * A snapshotted document that has since been mutated (content hash differs)
      is excluded, never silently served with its newer content. The pinned
      candidate set therefore only ever contains content byte-identical to
      snapshot time. Use :func:`snapshot_drift` to report what drifted.
    """

    expected = snapshot.content_hash_by_document()
    kept: list["LocalChunkRecord"] = []
    for chunk in chunks:
        document_id = str(chunk.metadata.get("document_id", ""))
        if document_id not in expected:
            continue
        if str(chunk.metadata.get("content_hash", "")) != expected[document_id]:
            continue
        kept.append(chunk)
    return kept


def snapshot_drift(
    chunks: Iterable["LocalChunkRecord"],
    snapshot: CorpusSnapshot,
) -> dict[str, list[str]]:
    """Report how the current corpus differs from a snapshot, honestly and by id.

    Returns sorted document-id lists under ``added`` (present now, not in the
    snapshot), ``removed`` (in the snapshot, gone now), and ``mutated`` (present
    in both but the content hash changed). ``mutated`` is drift that cannot be
    resurrected — only detected.
    """

    current: dict[str, str] = {}
    for chunk in chunks:
        document_id = str(chunk.metadata.get("document_id", ""))
        if document_id and document_id not in current:
            current[document_id] = str(chunk.metadata.get("content_hash", ""))

    recorded = snapshot.content_hash_by_document()
    added = sorted(document_id for document_id in current if document_id not in recorded)
    removed = sorted(document_id for document_id in recorded if document_id not in current)
    mutated = sorted(
        document_id
        for document_id, content_hash in current.items()
        if document_id in recorded and recorded[document_id] != content_hash
    )
    return {"added": added, "removed": removed, "mutated": mutated}
