"""Phase 3 deterministic query-routing smoke over local filings retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.embeddings import DEFAULT_VOYAGE_MODEL, VoyageEmbeddingProvider
from src.financial_rag.query import QueryPipeline
from src.financial_rag.retrieval import LocalDenseRetriever, load_local_retrieval_corpus
from src.financial_rag.settings import configured_secret, load_environment, project_root


DEFAULT_QUERY = "How have NVIDIA risk disclosures changed over the last year?"
DEFAULT_TICKER = "NVDA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3 local query smoke.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help=f"Question. Default: {DEFAULT_QUERY!r}")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Default ticker. Default: NVDA.")
    parser.add_argument("--top-k", type=int, default=5, help="Merged result count.")
    parser.add_argument("--per-subquery-k", type=int, default=5, help="Result count per subquery.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_environment()
    api_key = configured_secret("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit(
            "VOYAGE_API_KEY is required to embed query subqueries. "
            "This smoke reads cached chunks/vectors only and does not refetch SEC data."
        )

    chunks, embeddings = load_local_retrieval_corpus(root=project_root())
    if not chunks:
        raise SystemExit("No local chunks found under data/filings/chunks. Run Phase 1 ingestion first.")
    if not embeddings:
        raise SystemExit("No cached vectors found under data/vector_cache. Rerun Phase 1 with Voyage.")

    embedder = VoyageEmbeddingProvider(api_key=api_key, model=DEFAULT_VOYAGE_MODEL)
    retriever = LocalDenseRetriever(chunks=chunks, embeddings=embeddings, query_embedder=embedder)
    pipeline = QueryPipeline(retriever=retriever, chunks=chunks)
    result = pipeline.run(
        args.query,
        default_ticker=args.ticker,
        top_k=args.top_k,
        per_subquery_k=args.per_subquery_k,
    )

    routed = result.routed_query
    print(f"Query: {args.query}")
    print(f"Query type: {routed.query_type.value}")
    print(f"Filters: {routed.filters}")
    print(f"Subqueries: {len(result.subqueries)}")
    for subquery in result.subqueries:
        print(f"  {subquery.subquery_id}: {subquery.query}")
        print(f"    filters={subquery.filters}")
        print(f"    trace={subquery.trace}")

    print()
    print("Coverage:")
    for ticker, coverage in result.coverage.tickers.items():
        print(
            f"  {ticker}: chunks={coverage.chunk_count} forms={coverage.form_types} "
            f"roles={coverage.document_roles} ex99={coverage.exhibit_types}"
        )
        for gap in coverage.gaps:
            print(f"    gap: {gap}")

    print()
    print(f"Results returned: {len(result.results)}")
    for item in result.results:
        metadata = item.metadata
        parent_ids = item.parent_context.context_chunk_ids if item.parent_context else []
        print(f"{item.citation_label} score={item.dense_score:.4f} subquery={item.subquery_id}")
        print(f"  chunk_id: {item.chunk_id}")
        print(f"  url: {item.source_url}")
        print(
            "  metadata: "
            f"ticker={metadata.get('ticker', '')} "
            f"form_type={metadata.get('form_type', '')} "
            f"filing_date={metadata.get('filing_date', '')} "
            f"accession={metadata.get('accession_number', '')} "
            f"role={metadata.get('document_role', '')} "
            f"exhibit_type={metadata.get('exhibit_type', '')} "
            f"item={metadata.get('item_number', '')} "
            f"speaker={metadata.get('speaker_name', '')}"
        )
        print(f"  parent_context_chunks: {parent_ids}")
        print(f"  excerpt: {item.source_excerpt}")
        print()

    validation = result.citation_validation
    print(
        "Citation validation: "
        f"accepted={len(validation.accepted)} rejected={len(validation.rejected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
