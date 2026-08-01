"""Create, list, and inspect corpus snapshots (Phase 1 Stage 3).

A snapshot records the corpus composition at a moment in time so an eval pinned
to it reproduces even after new filings land. See
``src/financial_rag/corpus_snapshot.py`` for the honest scope: snapshots pin
state going forward; they do not resurrect content that already rolled forward.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.corpus_snapshot import (
    build_corpus_snapshot,
    list_corpus_snapshots,
    read_corpus_snapshot,
    snapshot_drift,
    write_corpus_snapshot,
)
from src.financial_rag.retrieval import load_local_retrieval_corpus
from src.financial_rag.settings import project_root
from src.financial_rag.storage import LocalRagStore


def _create() -> int:
    chunks, _ = load_local_retrieval_corpus(root=project_root())
    snapshot = build_corpus_snapshot(chunks)
    changed = write_corpus_snapshot(LocalRagStore(root=project_root()), snapshot)
    state = "recorded" if changed else "already present"
    print(f"Snapshot {snapshot.snapshot_id} {state}")
    print(f"  documents: {snapshot.document_count}")
    print(f"  chunks:    {snapshot.chunk_count}")
    print(f"  created:   {snapshot.created_at}")
    return 0


def _list() -> int:
    snapshots = list_corpus_snapshots(LocalRagStore(root=project_root()))
    if not snapshots:
        print("No snapshots recorded.")
        return 0
    for snapshot in snapshots:
        print(
            f"{snapshot.snapshot_id}  "
            f"docs={snapshot.document_count}  chunks={snapshot.chunk_count}  "
            f"created={snapshot.created_at}"
        )
    return 0


def _drift(snapshot_id: str) -> int:
    snapshot = read_corpus_snapshot(LocalRagStore(root=project_root()), snapshot_id)
    if snapshot is None:
        print(f"Snapshot not found: {snapshot_id}")
        return 1
    chunks, _ = load_local_retrieval_corpus(root=project_root())
    drift = snapshot_drift(chunks, snapshot)
    print(f"Drift vs {snapshot_id}:")
    for label in ("added", "removed", "mutated"):
        ids = drift[label]
        print(f"  {label} ({len(ids)}): {', '.join(ids) if ids else '-'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage corpus snapshots.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create", help="Snapshot the current live corpus and record it.")
    sub.add_parser("list", help="List recorded snapshots.")
    drift_parser = sub.add_parser("drift", help="Report how the live corpus differs from a snapshot.")
    drift_parser.add_argument("snapshot_id", help="Snapshot id to compare against.")
    args = parser.parse_args()

    if args.command == "create":
        return _create()
    if args.command == "list":
        return _list()
    if args.command == "drift":
        return _drift(args.snapshot_id)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
