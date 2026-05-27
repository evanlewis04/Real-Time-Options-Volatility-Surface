"""Phase 2 local dense retrieval smoke over cached SEC chunks and vectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.embeddings import DEFAULT_VOYAGE_MODEL, VoyageEmbeddingProvider
from src.financial_rag.retrieval import (
    LocalDenseRetriever,
    RetrievalFilters,
    load_local_retrieval_corpus,
)
from src.financial_rag.settings import configured_secret, load_environment, project_root
from src.financial_rag.synthesis import validate_citations


DEFAULT_QUERY = "What risks does NVIDIA describe?"
DEFAULT_TICKER = "NVDA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Phase 2 dense retrieval smoke.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help=f"Question. Default: {DEFAULT_QUERY!r}")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker filter. Default: NVDA.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    parser.add_argument("--form-type", default=None, help="Optional form type filter, e.g. 10-K.")
    parser.add_argument("--accession", default=None, help="Optional accession-number filter.")
    parser.add_argument("--document-role", default=None, help="Optional primary/exhibit filter.")
    parser.add_argument("--exhibit-type", default=None, help="Optional EX-99 exhibit type filter.")
    parser.add_argument("--item-number", default=None, help="Optional SEC item number filter, e.g. 1A.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_environment()
    api_key = configured_secret("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit(
            "VOYAGE_API_KEY is required to embed the retrieval query. "
            "This smoke reads cached chunks/vectors only and does not refetch SEC data."
        )

    chunks, embeddings = load_local_retrieval_corpus(root=project_root())
    if not chunks:
        raise SystemExit("No local chunks found under data/filings/chunks. Run Phase 1 ingestion first.")
    if not embeddings:
        raise SystemExit(
            "No cached vectors found under data/vector_cache. Rerun Phase 1 with VOYAGE_API_KEY first."
        )

    embedder = VoyageEmbeddingProvider(api_key=api_key, model=DEFAULT_VOYAGE_MODEL)
    retriever = LocalDenseRetriever(chunks=chunks, embeddings=embeddings, query_embedder=embedder)
    filters = RetrievalFilters(
        ticker=args.ticker,
        form_type=args.form_type,
        accession=args.accession,
        document_role=args.document_role,
        exhibit_type=args.exhibit_type,
        item_number=args.item_number,
    )
    results = retriever.search(query=args.query, top_k=args.top_k, filters=filters)

    print(f"Query: {args.query}")
    print(f"Chunks loaded: {len(chunks)}")
    print(f"Embeddings loaded: {len(embeddings)}")
    print(f"Results returned: {len(results)}")
    print()
    for result in results:
        metadata = result.metadata
        print(f"{result.citation_label} score={result.dense_score:.4f} chunk_id={result.chunk_id}")
        print(f"  url: {result.source_url}")
        print(
            "  metadata: "
            f"ticker={metadata.get('ticker', '')} "
            f"form_type={metadata.get('form_type', '')} "
            f"filing_date={metadata.get('filing_date', '')} "
            f"accession={metadata.get('accession_number', '')} "
            f"role={metadata.get('document_role', '')} "
            f"exhibit_type={metadata.get('exhibit_type', '')} "
            f"item={metadata.get('item_number', '')} "
            f"section={metadata.get('section_path', '')}"
        )
        print(f"  excerpt: {result.source_excerpt}")
        print()

    labels = [result.citation_label for result in results]
    validation = validate_citations(labels, results)
    print(
        "Citation validation: "
        f"accepted={len(validation.accepted)} rejected={len(validation.rejected)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
