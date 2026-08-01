"""Phase 1 SEC ingestion-to-embedding smoke pipeline.

The script downloads one ticker's latest 10-K, latest 10-Q, recent 8-K primary
documents, and EX-99 exhibits; parses readable text; creates deterministic
chunks; and embeds them with Voyage AI when `VOYAGE_API_KEY` is configured.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from src.financial_rag.chunking import chunk_document
from src.financial_rag.embeddings import (
    DEFAULT_VOYAGE_MODEL,
    EmbeddingCache,
    VoyageEmbeddingProvider,
    embedding_setup_message,
)
from src.financial_rag.ingestion.pinned_filings import PinnedFiling, cik_from_accession
from src.financial_rag.ingestion.sec_client import (
    SECClient,
    ExhibitRecord,
    FilingRecord,
    build_document_id,
    cik_padded,
    filing_primary_document_url,
    infer_exhibit_type,
    parse_filing_index_exhibits,
    recent_filing_records,
    sec_archive_base_url,
    select_phase1_filings,
)
from src.financial_rag.models import DocumentChunk, FilingMetadata
from src.financial_rag.parsing import classify_ex99_exhibit, extract_readable_text
from src.financial_rag.settings import configured_secret, load_environment, project_root
from src.financial_rag.storage import LocalRagStore


DEFAULT_TICKER = "NVDA"
DEFAULT_RECENT_8K_LIMIT = 5
DEFAULT_EMBED_BATCH_SIZE = 64


@dataclass
class SmokeCounts:
    fetched: int = 0
    already_present: int = 0
    parsed: int = 0
    parse_already_present: int = 0
    chunks_written: int = 0
    chunks_already_present: int = 0
    embedded: int = 0
    embeddings_already_present: int = 0
    embedding_skipped: int = 0


@dataclass(frozen=True)
class DownloadTarget:
    filing: FilingRecord
    document_name: str
    source_url: str
    document_role: str
    exhibit_type: str = ""
    description: str = ""


def run_smoke(
    *,
    ticker: str,
    recent_8k_limit: int,
    sec_delay: float,
    embed_batch_size: int,
    root: Path,
    skip_embeddings: bool = False,
) -> SmokeCounts:
    load_environment()
    sec_user_agent = configured_secret("SEC_USER_AGENT")
    if not sec_user_agent:
        raise SystemExit(
            "SEC_USER_AGENT must be configured in .env before using SEC EDGAR. "
            "Use a real contact string such as 'Your Name your.email@example.com'."
        )

    store = LocalRagStore(root=root)
    client = SECClient(user_agent=sec_user_agent, delay_seconds=sec_delay)
    company = client.lookup_company(ticker)
    submissions = client.fetch_company_submissions(company.cik)
    records = recent_filing_records(submissions)
    selection = select_phase1_filings(records, recent_8k_limit=recent_8k_limit)
    targets = discover_download_targets(client, company.cik, selection)
    counts = SmokeCounts()
    all_chunks: list[DocumentChunk] = []

    for target in targets:
        all_chunks.extend(
            ingest_target(
                store=store,
                client=client,
                ticker=company.ticker,
                cik=company.cik,
                company_name=company.company_name,
                target=target,
                counts=counts,
            )
        )

    if skip_embeddings:
        counts.embedding_skipped = len(all_chunks)
    else:
        embed_chunks(
            store=store,
            chunks=all_chunks,
            counts=counts,
            batch_size=embed_batch_size,
        )
    return counts


def run_pinned_filings(
    *,
    pins: tuple[PinnedFiling, ...],
    sec_delay: float,
    embed_batch_size: int,
    root: Path,
    skip_embeddings: bool = False,
) -> SmokeCounts:
    """Fetch, parse, chunk, and embed a fixed set of accession-pinned filings.

    Each pin is fetched under the CIK derived from its accession number (not the
    ticker->CIK lookup) and tagged under the pin's ticker, so eval-critical
    filings are present regardless of latest-N drift or decoy ticker mappings.
    """

    load_environment()
    sec_user_agent = configured_secret("SEC_USER_AGENT")
    if not sec_user_agent:
        raise SystemExit(
            "SEC_USER_AGENT must be configured in .env before using SEC EDGAR. "
            "Use a real contact string such as 'Your Name your.email@example.com'."
        )

    store = LocalRagStore(root=root)
    client = SECClient(user_agent=sec_user_agent, delay_seconds=sec_delay)
    counts = SmokeCounts()
    all_chunks: list[DocumentChunk] = []

    for pin in pins:
        cik = cik_from_accession(pin.accession_number)
        submissions = client.fetch_company_submissions(cik)
        company_name = str(submissions.get("name", ""))
        filing = find_filing_by_accession(recent_filing_records(submissions), pin.accession_number)
        for target in discover_pinned_targets(client, cik, filing):
            all_chunks.extend(
                ingest_target(
                    store=store,
                    client=client,
                    ticker=pin.ticker,
                    cik=cik_padded(cik),
                    company_name=company_name,
                    target=target,
                    counts=counts,
                )
            )

    if skip_embeddings:
        counts.embedding_skipped = len(all_chunks)
    else:
        embed_chunks(
            store=store,
            chunks=all_chunks,
            counts=counts,
            batch_size=embed_batch_size,
        )
    return counts


def find_filing_by_accession(
    records: tuple[FilingRecord, ...], accession_number: str
) -> FilingRecord:
    for record in records:
        if record.accession_number == accession_number:
            return record
    raise SystemExit(
        f"Pinned accession {accession_number} was not found in the filer's recent "
        "submissions. SEC EDGAR's submissions file lists roughly the latest 1,000 "
        "filings; a pin older than that window needs the paginated submissions index."
    )


def discover_pinned_targets(
    client: SECClient,
    cik: int | str,
    filing: FilingRecord,
) -> list[DownloadTarget]:
    """Download targets for a single pinned filing: primary doc plus EX-99 for 8-Ks."""

    targets = [
        DownloadTarget(
            filing=filing,
            document_name=filing.primary_document,
            source_url=filing_primary_document_url(cik, filing),
            document_role="primary",
            description=filing.description,
        )
    ]
    if filing.form == "8-K":
        base_url = sec_archive_base_url(cik, filing.accession_number)
        index_html = client.fetch_filing_index(cik, filing.accession_number)
        for exhibit in parse_filing_index_exhibits(index_html, base_url=base_url):
            targets.append(exhibit_download_target(filing, exhibit))
    return targets


def ingest_target(
    *,
    store: LocalRagStore,
    client: SECClient,
    ticker: str,
    cik: str,
    company_name: str,
    target: DownloadTarget,
    counts: SmokeCounts,
) -> list[DocumentChunk]:
    """Fetch, parse, and chunk one download target, updating ``counts`` in place."""

    metadata, created = fetch_raw_document(
        store=store,
        client=client,
        ticker=ticker,
        cik=cik,
        company_name=company_name,
        target=target,
    )
    if created:
        counts.fetched += 1
    else:
        counts.already_present += 1

    if parse_document(store=store, metadata=metadata):
        counts.parsed += 1
    else:
        counts.parse_already_present += 1

    chunks, chunks_created = chunk_parsed_document(store=store, metadata=metadata)
    if chunks_created:
        counts.chunks_written += len(chunks)
    else:
        counts.chunks_already_present += len(chunks)
    return chunks


def discover_download_targets(
    client: SECClient,
    cik: str,
    selection: object,
) -> list[DownloadTarget]:
    targets: list[DownloadTarget] = []
    for filing in (selection.latest_10k, selection.latest_10q):
        if filing is not None:
            targets.append(
                DownloadTarget(
                    filing=filing,
                    document_name=filing.primary_document,
                    source_url=filing_primary_document_url(cik, filing),
                    document_role="primary",
                    description=filing.description,
                )
            )

    for filing in selection.recent_8ks:
        targets.append(
            DownloadTarget(
                filing=filing,
                document_name=filing.primary_document,
                source_url=filing_primary_document_url(cik, filing),
                document_role="primary",
                description=filing.description,
            )
        )
        base_url = sec_archive_base_url(cik, filing.accession_number)
        index_html = client.fetch_filing_index(cik, filing.accession_number)
        for exhibit in parse_filing_index_exhibits(index_html, base_url=base_url):
            targets.append(exhibit_download_target(filing, exhibit))
    return targets


def exhibit_download_target(filing: FilingRecord, exhibit: ExhibitRecord) -> DownloadTarget:
    description = exhibit.description or exhibit.exhibit_type
    return DownloadTarget(
        filing=filing,
        document_name=exhibit.document,
        source_url=exhibit.url,
        document_role="exhibit",
        exhibit_type=infer_exhibit_type(exhibit.document, description),
        description=description,
    )


def fetch_raw_document(
    *,
    store: LocalRagStore,
    client: SECClient,
    ticker: str,
    cik: str,
    company_name: str,
    target: DownloadTarget,
) -> tuple[FilingMetadata, bool]:
    raw_path = store.raw_path(ticker, target.filing.accession_number, target.document_name)
    if raw_path.exists():
        content = raw_path.read_bytes()
        created = False
    else:
        content = client.fetch_document(target.source_url)
        created = store.write_bytes_once(raw_path, content).created
    content_hash = hashlib.sha256(content).hexdigest()
    exhibit_type = target.exhibit_type
    if target.document_role == "exhibit":
        exhibit_type = classify_ex99_exhibit(
            filename=target.document_name,
            description=target.description,
            text=extract_readable_text(content)[:4000],
        )
    document_id = build_document_id(
        ticker=ticker,
        accession_number=target.filing.accession_number,
        document_role=target.document_role,
        document_name=target.document_name,
    )
    metadata = FilingMetadata(
        document_id=document_id,
        ticker=ticker,
        cik=cik,
        company_name=company_name,
        accession_number=target.filing.accession_number,
        form_type=target.filing.form if target.document_role == "primary" else "EX-99",
        filing_date=target.filing.filing_date,
        report_date=target.filing.report_date,
        source_url=target.source_url,
        local_path=str(raw_path.resolve()),
        document_role=target.document_role,
        exhibit_type=exhibit_type,
        document_name=target.document_name,
        description=target.description,
        content_hash=content_hash,
    )
    store.upsert_manifest(
        store.raw_dir / "manifest.jsonl",
        key="document_id",
        record=metadata.to_dict(),
    )
    return metadata, created


def parse_document(*, store: LocalRagStore, metadata: FilingMetadata) -> bool:
    parsed_path = store.parsed_path(metadata.document_id)
    raw_content = Path(metadata.local_path).read_bytes()
    text = extract_readable_text(raw_content)
    result = store.write_text_once(parsed_path, text)
    parsed_record = {
        **metadata.to_dict(),
        "parsed_path": str(parsed_path.resolve()),
        "character_count": len(text),
    }
    store.upsert_manifest(
        store.parsed_dir / "manifest.jsonl",
        key="document_id",
        record=parsed_record,
    )
    return result.created


def chunk_parsed_document(
    *,
    store: LocalRagStore,
    metadata: FilingMetadata,
) -> tuple[list[DocumentChunk], bool]:
    parsed_path = store.parsed_path(metadata.document_id)
    text = parsed_path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_document(text, metadata)
    result = store.write_jsonl_once(
        store.chunks_path(metadata.document_id),
        [chunk.to_dict() for chunk in chunks],
    )
    for chunk in chunks:
        store.upsert_manifest(
            store.chunks_dir / "manifest.jsonl",
            key="chunk_id",
            record={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "ticker": chunk.ticker,
                "cik": chunk.cik,
                "accession_number": chunk.accession_number,
                "form_type": chunk.form_type,
                "filing_date": chunk.filing_date,
                "source_url": chunk.source_url,
                "local_path": chunk.local_path,
                "document_role": chunk.document_role,
                "exhibit_type": chunk.exhibit_type,
                "start_offset": chunk.start_offset,
                "end_offset": chunk.end_offset,
                "token_count": chunk.token_count,
                "section_path": chunk.section_path,
                "item_number": chunk.item_number,
                "speaker_name": chunk.speaker_name,
                "speaker_role": chunk.speaker_role,
            },
        )
    return chunks, result.created


def embed_chunks(
    *,
    store: LocalRagStore,
    chunks: list[DocumentChunk],
    counts: SmokeCounts,
    batch_size: int,
) -> None:
    api_key = configured_secret("VOYAGE_API_KEY")
    if not api_key:
        counts.embedding_skipped = len(chunks)
        print(embedding_setup_message())
        return

    try:
        provider = VoyageEmbeddingProvider(api_key=api_key, model=DEFAULT_VOYAGE_MODEL)
    except RuntimeError as exc:
        counts.embedding_skipped = len(chunks)
        print(str(exc))
        return
    cache = EmbeddingCache(store=store, model=provider.model)
    uncached = [chunk for chunk in chunks if not cache.is_cached(chunk)]
    counts.embeddings_already_present = len(chunks) - len(uncached)
    for start in range(0, len(uncached), batch_size):
        batch = uncached[start : start + batch_size]
        vectors = provider.embed_texts([chunk.chunk_text for chunk in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            if cache.write(chunk, vector):
                counts.embedded += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 filings RAG smoke pipeline.")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker to ingest. Default: NVDA.")
    parser.add_argument(
        "--recent-8k-limit",
        type=int,
        default=DEFAULT_RECENT_8K_LIMIT,
        help="Number of recent exact-form 8-K filings to include.",
    )
    parser.add_argument(
        "--sec-delay",
        type=float,
        default=0.25,
        help="Minimum seconds between SEC requests.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help="Voyage embedding batch size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = run_smoke(
        ticker=args.ticker,
        recent_8k_limit=args.recent_8k_limit,
        sec_delay=args.sec_delay,
        embed_batch_size=args.embed_batch_size,
        root=project_root(),
    )
    print(f"Fetched raw documents: {counts.fetched}")
    print(f"Already present/skipped raw documents: {counts.already_present}")
    print(f"Parsed documents written: {counts.parsed}")
    print(f"Parsed documents already present: {counts.parse_already_present}")
    print(f"Chunks written: {counts.chunks_written}")
    print(f"Chunks already present: {counts.chunks_already_present}")
    print(f"Embeddings written: {counts.embedded}")
    print(f"Embeddings already present: {counts.embeddings_already_present}")
    print(f"Embeddings skipped: {counts.embedding_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
