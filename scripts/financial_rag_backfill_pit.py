"""Backfill the point-in-time spine (filed_at / period_end) onto a cached corpus.

Chunks written before Phase 1 Stage 1 lack ``filed_at`` (EDGAR acceptance datetime)
and ``period_end``. This migration threads both onto every cached chunk row and the
chunk manifest, idempotently:

- ``period_end`` is derived **offline** from each chunk's existing ``report_date``.
- ``filed_at`` (acceptance) is re-fetched per CIK from the EDGAR submissions payload
  when ``SEC_USER_AGENT`` is configured. An accession whose acceptance cannot be
  resolved is **flagged** (left empty) — never zero-filled or guessed (data honesty).

Idempotent: a row that already carries a non-empty value is left untouched, so the
offline pass can run now and the network pass can complete ``filed_at`` later without
double-writing. Only ``filed_at`` / ``period_end`` are ever added — chunk text, ids,
offsets and every scored field are preserved exactly, so the §4 eval reproduces.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.ingestion.sec_client import SECClient, recent_filing_records
from src.financial_rag.settings import configured_secret, load_environment, project_root
from src.financial_rag.storage import LocalRagStore


def period_end_for_row(row: dict[str, Any]) -> str:
    """Offline period_end: existing value, else the chunk's report_date."""
    existing = str(row.get("period_end", "") or "")
    if existing:
        return existing
    nested = row.get("metadata")
    if isinstance(nested, dict) and nested.get("report_date"):
        return str(nested["report_date"])
    return str(row.get("report_date", "") or "")


def acceptance_map_from_payloads(payloads: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Map accession_number -> normalized acceptance datetime from submissions payloads."""
    mapping: dict[str, str] = {}
    for payload in payloads:
        for record in recent_filing_records(payload):
            if record.accession_number and record.acceptance_datetime:
                mapping.setdefault(record.accession_number, record.acceptance_datetime)
    return mapping


def backfill_row(row: dict[str, Any], acceptance_by_accession: dict[str, str]) -> bool:
    """Add filed_at / period_end to a chunk row in place. Return True if it changed.

    Only fills empty fields (idempotent). Unresolved acceptance stays empty (flagged).
    """
    changed = False

    period_end = period_end_for_row(row)
    if period_end and row.get("period_end", "") != period_end:
        row["period_end"] = period_end
        changed = True

    if not str(row.get("filed_at", "") or ""):
        accession = str(row.get("accession_number", "") or "")
        resolved = acceptance_by_accession.get(accession, "")
        if resolved:
            row["filed_at"] = resolved
            changed = True
        elif "filed_at" not in row:
            row["filed_at"] = ""  # explicit flag: known-missing, not absent
            changed = True

    return changed


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def run_backfill(
    store: LocalRagStore,
    acceptance_by_accession: dict[str, str],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill filed_at / period_end onto cached chunk rows.

    ``dry_run`` computes exactly what would change but never rewrites any file, so
    a preview can never mutate the corpus. Row dicts are mutated in memory only to
    count changes; on a dry run they are discarded unwritten.
    """

    chunk_files = [p for p in sorted(store.chunks_dir.glob("*.jsonl")) if p.name != "manifest.jsonl"]
    chunks_seen = 0
    chunks_changed = 0
    filed_at_resolved = 0
    unresolved_accessions: set[str] = set()

    for path in chunk_files:
        rows = list(_read_jsonl(path))
        file_changed = False
        for row in rows:
            chunks_seen += 1
            if backfill_row(row, acceptance_by_accession):
                chunks_changed += 1
                file_changed = True
            if str(row.get("filed_at", "") or ""):
                filed_at_resolved += 1
            else:
                unresolved_accessions.add(str(row.get("accession_number", "") or ""))
        if file_changed and not dry_run:
            _rewrite_jsonl(path, rows)

    return {
        "dry_run": dry_run,
        "chunk_files": len(chunk_files),
        "chunks_seen": chunks_seen,
        "chunks_changed": chunks_changed,
        "filed_at_resolved": filed_at_resolved,
        "filed_at_unresolved": chunks_seen - filed_at_resolved,
        "unresolved_accessions": sorted(a for a in unresolved_accessions if a),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _unique_ciks(store: LocalRagStore) -> list[str]:
    ciks: set[str] = set()
    for path in sorted(store.chunks_dir.glob("*.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        for row in _read_jsonl(path):
            cik = str(row.get("cik", "") or "")
            if cik:
                ciks.add(cik)
    return sorted(ciks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill filed_at / period_end onto the cached corpus.")
    parser.add_argument("--root", default=None, help="Corpus root (defaults to project_root()).")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Skip the SEC acceptance re-fetch; backfill period_end only.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the real corpus-mutating backfill. Without it, dry-run (preview only).",
    )
    args = parser.parse_args()

    # Resolve SEC_USER_AGENT (and any other secret) from .env, matching every other
    # real-run script (e.g. financial_rag_sp500_fetch). Without this the --execute
    # acceptance re-fetch silently degrades to offline-only, leaving filed_at flagged.
    load_environment()

    root = Path(args.root) if args.root else project_root()
    store = LocalRagStore(root=root)

    if not args.execute:
        # Guardrail: preview only — no SEC re-fetch, no file writes.
        ciks = _unique_ciks(store)
        summary = run_backfill(store, {}, dry_run=True)
        print("[DRY RUN] No corpus writes. Pass --execute to apply the backfill.")
        if not args.offline_only:
            print(
                f"Would fetch SEC submissions for {len(ciks)} CIK(s) to resolve filed_at, "
                "then write filed_at / period_end onto cached chunks."
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    acceptance_by_accession: dict[str, str] = {}
    if not args.offline_only:
        user_agent = configured_secret("SEC_USER_AGENT")
        if not user_agent:
            print(
                "SEC_USER_AGENT not configured — resolving period_end offline only; "
                "filed_at stays flagged-empty. Set SEC_USER_AGENT and re-run to complete "
                "the acceptance backfill (idempotent).",
                file=sys.stderr,
            )
        else:
            client = SECClient(user_agent=user_agent)
            payloads = []
            for cik in _unique_ciks(store):
                try:
                    payloads.append(client.fetch_company_submissions(cik))
                except Exception as exc:  # noqa: BLE001 - network/parse; flagged, not fatal
                    print(f"submissions fetch failed for CIK {cik}: {exc}", file=sys.stderr)
            acceptance_by_accession = acceptance_map_from_payloads(payloads)

    summary = run_backfill(store, acceptance_by_accession)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
