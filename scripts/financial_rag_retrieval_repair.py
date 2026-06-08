"""Repair local filings retrieval by rebuilding chunks and optional embeddings.

Default behavior is cache-only: no SEC request is made unless `--fetch-sec` is
provided. Raw filings, parsed text, chunks, vectors, and reports stay under
ignored local data/artifact paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.financial_rag_ingest import DEFAULT_EMBED_BATCH_SIZE, run_smoke
from src.financial_rag.chunking import chunk_document
from src.financial_rag.embeddings import (
    DEFAULT_VOYAGE_MODEL,
    EmbeddingCache,
    VoyageEmbeddingProvider,
    embedding_setup_message,
)
from src.financial_rag.models import DocumentChunk, EmbeddingRecord, FilingMetadata
from src.financial_rag.parsing import extract_readable_text
from src.financial_rag.settings import configured_secret, load_environment, project_root
from src.financial_rag.storage.local_store import LocalRagStore, read_jsonl


DEFAULT_TICKERS = "NVDA,AMD,MSFT,AAPL,JPM,XOM,INTC,GOOGL,META,AMZN,BAC,GS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair/rebuild local financial RAG retrieval cache.")
    parser.add_argument("--tickers", default=DEFAULT_TICKERS, help="Comma-separated tickers to repair.")
    parser.add_argument("--recent-8k-limit", type=int, default=3, help="Recent 8-K count when --fetch-sec is used.")
    parser.add_argument("--sec-delay", type=float, default=0.25, help="SEC request delay when --fetch-sec is used.")
    parser.add_argument("--fetch-sec", action="store_true", help="Explicitly fetch missing/latest SEC filings first.")
    parser.add_argument("--embed", action="store_true", help="Generate missing Voyage embeddings after chunk rebuild.")
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)
    parser.add_argument("--max-embed-chunks", type=int, default=None, help="Limit embeddings written in one resumable run.")
    parser.add_argument(
        "--prune-stale-embeddings",
        action="store_true",
        help="Delete vector-cache files whose chunk IDs are no longer present after rechunking.",
    )
    parser.add_argument(
        "--summary-output",
        default="artifacts/rag_eval/retrieval_repair_summary.json",
        help="Ignored JSON summary path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = project_root()
    load_environment()
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if args.fetch_sec:
        for ticker in tickers:
            print(f"Fetching SEC cache for {ticker}")
            run_smoke(
                ticker=ticker,
                recent_8k_limit=args.recent_8k_limit,
                sec_delay=args.sec_delay,
                embed_batch_size=args.embed_batch_size,
                root=root,
                skip_embeddings=not args.embed,
            )

    store = LocalRagStore(root=root)
    raw_rows = [
        row
        for row in read_jsonl(store.raw_dir / "manifest.jsonl")
        if str(row.get("ticker", "")).upper() in set(tickers)
    ]
    if not raw_rows:
        print("No cached raw filings matched the requested tickers. Rerun with --fetch-sec to populate SEC cache.")
        return 1

    chunks_by_doc: dict[str, list[DocumentChunk]] = {}
    parsed_count = 0
    chunk_count = 0
    for row in raw_rows:
        metadata = _metadata_from_row(row)
        parsed_text = extract_readable_text(Path(metadata.local_path).read_bytes())
        parsed_path = store.parsed_path(metadata.document_id)
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path.write_text(parsed_text, encoding="utf-8")
        store.upsert_manifest(
            store.parsed_dir / "manifest.jsonl",
            key="document_id",
            record={**metadata.to_dict(), "parsed_path": str(parsed_path.resolve()), "character_count": len(parsed_text)},
        )
        parsed_count += 1

        chunks = chunk_document(parsed_text, metadata)
        chunks_by_doc[metadata.document_id] = chunks
        chunk_path = store.chunks_path(metadata.document_id)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_path.write_text("".join(json.dumps(chunk.to_dict(), sort_keys=True) + "\n" for chunk in chunks), encoding="utf-8")
        chunk_count += len(chunks)

    _rebuild_chunk_manifest(store)
    if args.embed:
        _rebuild_embedding_manifest(store)
    embedded_count = _embed_chunks(store, [chunk for chunks in chunks_by_doc.values() for chunk in chunks], args)
    embedding_status = _embedding_status(store)
    stale_pruned = _prune_stale_embeddings(store, embedding_status["stale_embedding_ids"]) if args.prune_stale_embeddings else 0
    if stale_pruned:
        embedding_status = _embedding_status(store)
    summary = {
        "tickers": tickers,
        "fetch_sec": args.fetch_sec,
        "documents_rebuilt": parsed_count,
        "chunks_rebuilt": chunk_count,
        "embeddings_written": embedded_count,
        "stale_embeddings_pruned": stale_pruned,
        "current_chunk_count": embedding_status["current_chunk_count"],
        "current_embedding_count": embedding_status["current_embedding_count"],
        "missing_current_embedding_count": embedding_status["missing_current_embedding_count"],
        "stale_embedding_count": embedding_status["stale_embedding_count"],
        "chunker": "sec-aware",
    }
    output_path = root / args.summary_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Documents rebuilt: {parsed_count}")
    print(f"Chunks rebuilt: {chunk_count}")
    print(f"Embeddings written: {embedded_count}")
    print(f"Current chunks missing embeddings: {embedding_status['missing_current_embedding_count']}")
    print(f"Stale embeddings: {embedding_status['stale_embedding_count']}")
    print(f"Stale embeddings pruned: {stale_pruned}")
    print(f"Summary: {output_path}")
    return 0


def _metadata_from_row(row: dict[str, object]) -> FilingMetadata:
    return FilingMetadata(
        document_id=str(row.get("document_id", "")),
        ticker=str(row.get("ticker", "")),
        cik=str(row.get("cik", "")),
        company_name=str(row.get("company_name", "")),
        accession_number=str(row.get("accession_number", "")),
        form_type=str(row.get("form_type", "")),
        filing_date=str(row.get("filing_date", "")),
        report_date=str(row.get("report_date", "")),
        source_url=str(row.get("source_url", "")),
        local_path=str(row.get("local_path", "")),
        document_role=str(row.get("document_role", "")),
        exhibit_type=str(row.get("exhibit_type", "")),
        document_name=str(row.get("document_name", "")),
        description=str(row.get("description", "")),
        content_hash=str(row.get("content_hash", "")),
    )


def _rebuild_chunk_manifest(store: LocalRagStore) -> None:
    rows = []
    for path in sorted(store.chunks_dir.glob("*.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        for row in read_jsonl(path):
            rows.append(
                {
                    "chunk_id": row.get("chunk_id", ""),
                    "document_id": row.get("document_id", ""),
                    "ticker": row.get("ticker", ""),
                    "cik": row.get("cik", ""),
                    "accession_number": row.get("accession_number", ""),
                    "form_type": row.get("form_type", ""),
                    "filing_date": row.get("filing_date", ""),
                    "source_url": row.get("source_url", ""),
                    "local_path": row.get("local_path", ""),
                    "document_role": row.get("document_role", ""),
                    "exhibit_type": row.get("exhibit_type", ""),
                    "start_offset": row.get("start_offset", 0),
                    "end_offset": row.get("end_offset", 0),
                    "token_count": row.get("token_count", 0),
                    "section_path": row.get("section_path", ""),
                    "item_number": row.get("item_number", ""),
                    "speaker_name": row.get("speaker_name", ""),
                    "speaker_role": row.get("speaker_role", ""),
                }
            )
    manifest_path = store.chunks_dir / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _embed_chunks(store: LocalRagStore, chunks: list[DocumentChunk], args: argparse.Namespace) -> int:
    if not args.embed:
        return 0
    api_key = configured_secret("VOYAGE_API_KEY")
    if not api_key:
        print(embedding_setup_message())
        return 0
    provider = VoyageEmbeddingProvider(api_key=api_key, model=DEFAULT_VOYAGE_MODEL)
    cache = EmbeddingCache(store=store, model=provider.model)
    uncached = [chunk for chunk in chunks if not cache.is_cached(chunk)]
    if args.max_embed_chunks is not None:
        uncached = uncached[: max(args.max_embed_chunks, 0)]
    written = 0
    for start in range(0, len(uncached), args.embed_batch_size):
        batch = uncached[start : start + args.embed_batch_size]
        vectors = provider.embed_texts([chunk.chunk_text for chunk in batch])
        for chunk, vector in zip(batch, vectors, strict=True):
            if _write_embedding_file(store, chunk, vector, provider.model):
                written += 1
    if written:
        _rebuild_embedding_manifest(store)
    return written


def _write_embedding_file(store: LocalRagStore, chunk: DocumentChunk, embedding: list[float], model: str) -> bool:
    path = store.embedding_path(chunk.chunk_id)
    if path.exists():
        return False
    record = EmbeddingRecord(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        model=model,
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
    path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return True


def _rebuild_embedding_manifest(store: LocalRagStore) -> None:
    rows = []
    for path in sorted(store.vector_cache_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("chunk_id"):
            rows.append(payload)
    (store.vector_cache_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _embedding_status(store: LocalRagStore) -> dict[str, object]:
    current_chunk_ids = _current_chunk_ids(store)
    embedding_ids = _embedding_ids(store)
    missing = sorted(current_chunk_ids - embedding_ids)
    stale = sorted(embedding_ids - current_chunk_ids)
    return {
        "current_chunk_count": len(current_chunk_ids),
        "current_embedding_count": len(current_chunk_ids & embedding_ids),
        "missing_current_embedding_count": len(missing),
        "stale_embedding_count": len(stale),
        "missing_current_embedding_ids": missing,
        "stale_embedding_ids": stale,
    }


def _current_chunk_ids(store: LocalRagStore) -> set[str]:
    chunk_ids: set[str] = set()
    for path in sorted(store.chunks_dir.glob("*.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        for row in read_jsonl(path):
            chunk_id = str(row.get("chunk_id", ""))
            if chunk_id:
                chunk_ids.add(chunk_id)
    return chunk_ids


def _embedding_ids(store: LocalRagStore) -> set[str]:
    ids: set[str] = set()
    for path in sorted(store.vector_cache_dir.glob("*.json")):
        ids.add(path.stem)
    return ids


def _prune_stale_embeddings(store: LocalRagStore, stale_embedding_ids: object) -> int:
    pruned = 0
    for chunk_id in stale_embedding_ids if isinstance(stale_embedding_ids, list) else []:
        path = store.embedding_path(str(chunk_id))
        if path.exists():
            path.unlink()
            pruned += 1
    manifest_path = store.vector_cache_dir / "manifest.jsonl"
    if manifest_path.exists():
        current_ids = _current_chunk_ids(store)
        rows = [row for row in read_jsonl(manifest_path) if str(row.get("chunk_id", "")) in current_ids]
        manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return pruned


if __name__ == "__main__":
    raise SystemExit(main())
