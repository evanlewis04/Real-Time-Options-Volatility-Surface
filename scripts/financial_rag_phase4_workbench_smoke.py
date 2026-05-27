"""Phase 4 smoke for local API/workbench helpers without SEC refetch."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.api import QueryRequest, build_local_api_service
from src.financial_rag.settings import project_root
from src.financial_rag.workbench import coverage_rows, evidence_rows


DEFAULT_QUERY = "How have NVIDIA risk disclosures changed over the last year?"


def main() -> int:
    service = build_local_api_service(root=project_root(), use_voyage=False)
    health = service.health()
    coverage = service.coverage(tickers=["NVDA"])
    payload = service.query(QueryRequest(question=DEFAULT_QUERY, ticker="NVDA", top_k=3, per_subquery_k=3))
    rows = evidence_rows(payload)
    coverage_table = coverage_rows(coverage)

    print(f"Health: {health['status']}")
    print(f"Chunks: {health['chunk_count']}")
    print(f"Embeddings: {health['embedding_count']}")
    print(f"Query type: {payload['routed_query']['query_type']}")
    print(f"Subqueries: {len(payload['subqueries'])}")
    print(f"Evidence rows: {len(rows)}")
    print(f"Coverage rows: {len(coverage_table)}")
    print(
        "Citation validation: "
        f"accepted={len(payload['citations']['accepted'])} "
        f"rejected={len(payload['citations']['rejected'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
