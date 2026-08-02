"""Batched, resumable fetch orchestration for many-ticker ingestion.

Phase 1 Stage 4 pulls the S&P 500 (~500 names × multiple filings each). A run that
long must survive interruption, so ``fetch_constituents`` drives a per-ticker
checkpoint: a ticker is recorded as completed only *after* its full ingest
succeeds, so a re-run skips completed names and re-runs the interrupted one. The
underlying document/parse/chunk writes are already idempotent
(``write_*_once`` / ``upsert_manifest``), so re-running the interrupted name
re-uses everything already on disk and never produces duplicates.

The actual per-ticker ingest is injected (``ingest_company``): this module owns
only the orchestration — skip-completed, checkpoint-record, and failure handling —
and stays independent of the fetch/parse/chunk engine so it is unit-testable
against a mocked SEC client with no live EDGAR call.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.financial_rag.ingestion.sec_client import CompanyRecord
from src.financial_rag.storage import LocalRagStore
from src.financial_rag.storage.local_store import read_jsonl

CHECKPOINT_KEY = "ticker"
STATUS_COMPLETED = "completed"

# ``ingest_company`` fetches, parses, and chunks one company's filings and returns
# its chunks. It raises on any fetch/parse failure — a raised exception is treated
# as an interruption: the ticker is left un-checkpointed so a re-run retries it.
IngestCompany = Callable[[CompanyRecord], list]


def load_completed_tickers(checkpoint_path: Path) -> set[str]:
    """Return the set of tickers already recorded ``completed`` in the checkpoint."""

    return {
        normalized
        for row in read_jsonl(checkpoint_path)
        if str(row.get("status", "")) == STATUS_COMPLETED
        and (normalized := str(row.get("ticker", "") or "").upper())
    }


@dataclass
class BatchFetchResult:
    """Summary of one batched fetch run."""

    total: int = 0
    skipped: int = 0
    completed: int = 0
    failed: int = 0
    chunks_written: int = 0
    # Names that a dry run would fetch (not yet complete). Stays 0 on a real run.
    pending: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)


def fetch_constituents(
    *,
    constituents: Sequence[CompanyRecord],
    store: LocalRagStore,
    checkpoint_path: Path,
    ingest_company: IngestCompany,
    continue_on_error: bool = False,
    dry_run: bool = False,
    logger: Callable[[str], None] = print,
) -> BatchFetchResult:
    """Fetch every constituent once, checkpointing after each success.

    Already-completed tickers (per ``checkpoint_path``) are skipped without any
    network call. On success the ticker is upserted into the checkpoint keyed by
    ticker (idempotent). On failure the ticker is recorded as a failure but never
    checkpointed as completed, so it re-runs next time; with
    ``continue_on_error=False`` (the default) the exception also propagates after
    it is recorded, modelling a hard interruption that stops the run.

    ``dry_run`` reports the plan (how many names would be fetched vs. already
    complete) without calling ``ingest_company`` or writing the checkpoint — zero
    corpus mutation and no network, so a real pull is always a deliberate opt-in.
    """

    completed = load_completed_tickers(checkpoint_path)
    result = BatchFetchResult(total=len(constituents))

    if dry_run:
        for company in constituents:
            if company.ticker.upper() in completed:
                result.skipped += 1
            else:
                result.pending += 1
        logger(
            f"[sp500] DRY RUN: would fetch {result.pending} name(s), "
            f"{result.skipped} already complete of {result.total}. "
            "No corpus writes. Re-run with --execute to perform the real pull."
        )
        return result

    for company in constituents:
        ticker = company.ticker.upper()
        if ticker in completed:
            result.skipped += 1
            continue

        try:
            chunks = ingest_company(company)
        except Exception as exc:  # noqa: BLE001 — any failure is a resumable interruption
            result.failed += 1
            result.failures.append((ticker, str(exc)))
            logger(f"[sp500] {ticker} FAILED: {exc}")
            if continue_on_error:
                continue
            raise

        chunk_count = len(chunks)
        result.chunks_written += chunk_count
        store.upsert_manifest(
            checkpoint_path,
            key=CHECKPOINT_KEY,
            record={
                "ticker": ticker,
                "cik": company.cik,
                "company_name": company.company_name,
                "status": STATUS_COMPLETED,
                "chunk_count": chunk_count,
            },
        )
        completed.add(ticker)
        result.completed += 1
        logger(f"[sp500] {ticker} done ({chunk_count} chunks)")

    return result
