"""Batched, resumable S&P 500 ingestion (Phase 1 Stage 4).

Reads the committed constituent list (``config/sp500_constituents.csv``) and
fetches each name's Phase 1 filing set (latest 10-K, latest 10-Q, recent 8-Ks +
EX-99) through the existing ingest engine, checkpointing after each completed
name so an interrupted run resumes where it stopped and never re-fetches a done
name. Respects the existing SEC rate-limit posture (``DEFAULT_SEC_DELAY_SECONDS``);
no new throttle knobs.

This is the *machinery*. The real 500-company pull is gated on ``SEC_USER_AGENT``
(the same secret as the Stage 1 backfill) and runs when it lands — this script is
then a one-command run.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.ingestion.batch_fetch import BatchFetchResult, fetch_constituents
from src.financial_rag.ingestion.constituents import parse_constituents_csv
from src.financial_rag.ingestion.sec_client import (
    CompanyRecord,
    DEFAULT_SEC_DELAY_SECONDS,
    SECClient,
    recent_filing_records,
    select_phase1_filings,
)
from src.financial_rag.models import DocumentChunk
from src.financial_rag.settings import configured_secret, load_environment, project_root
from src.financial_rag.storage import LocalRagStore

# The fetch/parse/chunk engine lives in the sibling ingest script; load it by path
# (scripts/ is not an importable package) rather than duplicating the pipeline.
_INGEST_SPEC = importlib.util.spec_from_file_location(
    "financial_rag_ingest", Path(__file__).with_name("financial_rag_ingest.py")
)
assert _INGEST_SPEC and _INGEST_SPEC.loader
ingest_engine = importlib.util.module_from_spec(_INGEST_SPEC)
# Register before exec: the engine's `from __future__ import annotations` dataclasses
# resolve field annotations via ``sys.modules[__name__]`` at class-creation time.
sys.modules[_INGEST_SPEC.name] = ingest_engine
_INGEST_SPEC.loader.exec_module(ingest_engine)

DEFAULT_CONSTITUENTS = Path("config") / "sp500_constituents.csv"
DEFAULT_RECENT_8K_LIMIT = 5
DEFAULT_EMBED_BATCH_SIZE = 64


def checkpoint_path(store: LocalRagStore) -> Path:
    """Resumable progress record, alongside the other filings artifacts (gitignored)."""

    return store.snapshots_dir.parent / "sp500_fetch_checkpoint.jsonl"


def make_ingest_company(
    *,
    client: SECClient,
    store: LocalRagStore,
    recent_8k_limit: int,
    embed_batch_size: int,
    skip_embeddings: bool,
):
    """Build the per-company ingest callable driven through the shared engine."""

    def ingest_company(company: CompanyRecord) -> list[DocumentChunk]:
        submissions = client.fetch_company_submissions(company.cik)
        records = recent_filing_records(submissions)
        selection = select_phase1_filings(records, recent_8k_limit=recent_8k_limit)
        targets = ingest_engine.discover_download_targets(client, company.cik, selection)
        counts = ingest_engine.SmokeCounts()
        chunks: list[DocumentChunk] = []
        for target in targets:
            chunks.extend(
                ingest_engine.ingest_target(
                    store=store,
                    client=client,
                    ticker=company.ticker,
                    cik=company.cik,
                    company_name=company.company_name,
                    target=target,
                    counts=counts,
                )
            )
        if not skip_embeddings:
            ingest_engine.embed_chunks(
                store=store, chunks=chunks, counts=counts, batch_size=embed_batch_size
            )
        return chunks

    return ingest_company


def run(
    *,
    constituents_path: Path,
    root: Path,
    recent_8k_limit: int,
    sec_delay: float,
    embed_batch_size: int,
    limit: int | None,
    skip_embeddings: bool,
) -> BatchFetchResult:
    load_environment()
    sec_user_agent = configured_secret("SEC_USER_AGENT")
    if not sec_user_agent:
        raise SystemExit(
            "SEC_USER_AGENT must be configured in .env before the real S&P 500 pull. "
            "The Stage 4 machinery and its tests run without it; only the live fetch needs it."
        )

    constituents = list(parse_constituents_csv(constituents_path.read_text(encoding="utf-8")))
    if limit is not None:
        constituents = constituents[:limit]

    store = LocalRagStore(root=root)
    client = SECClient(user_agent=sec_user_agent, delay_seconds=sec_delay)
    ingest_company = make_ingest_company(
        client=client,
        store=store,
        recent_8k_limit=recent_8k_limit,
        embed_batch_size=embed_batch_size,
        skip_embeddings=skip_embeddings,
    )
    return fetch_constituents(
        constituents=constituents,
        store=store,
        checkpoint_path=checkpoint_path(store),
        ingest_company=ingest_company,
        continue_on_error=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched, resumable S&P 500 ingestion.")
    parser.add_argument(
        "--constituents",
        default=str(DEFAULT_CONSTITUENTS),
        help=f"Constituents CSV. Default: {DEFAULT_CONSTITUENTS}.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Fetch only the first N constituents."
    )
    parser.add_argument(
        "--recent-8k-limit",
        type=int,
        default=DEFAULT_RECENT_8K_LIMIT,
        help="Number of recent exact-form 8-K filings per name.",
    )
    parser.add_argument(
        "--sec-delay",
        type=float,
        default=DEFAULT_SEC_DELAY_SECONDS,
        help="Minimum seconds between SEC requests (existing rate-limit posture).",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE, help="Voyage batch size."
    )
    parser.add_argument("--skip-embeddings", action="store_true", help="Fetch/parse/chunk only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        constituents_path=project_root() / args.constituents,
        root=project_root(),
        recent_8k_limit=args.recent_8k_limit,
        sec_delay=args.sec_delay,
        embed_batch_size=args.embed_batch_size,
        limit=args.limit,
        skip_embeddings=args.skip_embeddings,
    )
    print(f"Constituents considered: {result.total}")
    print(f"Skipped (already complete): {result.skipped}")
    print(f"Completed this run: {result.completed}")
    print(f"Chunks written: {result.chunks_written}")
    print(f"Failed (will retry next run): {result.failed}")
    for ticker, error in result.failures:
        print(f"  {ticker}: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
