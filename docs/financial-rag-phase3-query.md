# Financial RAG Phase 3 Query Sophistication

Phase 3 adds deterministic query routing and local retrieval orchestration on
top of the Phase 2 dense retriever. It stays backend/library-first: no frontend,
agents, answer synthesis, Qdrant, paid transcript integrations, OpenAI calls,
Anthropic calls, or production evals.

## Command

```bash
.\venv\Scripts\python.exe scripts\financial_rag_phase3_query_smoke.py
```

Defaults:

- Query: `How have NVIDIA risk disclosures changed over the last year?`
- Default ticker: `NVDA`
- Merged top-k: `5`

## Prerequisites

- Local chunks under `data/filings/chunks/` from the Phase 1/2 pipeline.
- Cached Voyage vectors under `data/vector_cache/`.
- `VOYAGE_API_KEY` for embedding query subqueries.
- No SEC user-agent is needed because the script reads local cache only.

## Behavior

- Classifies the query with deterministic rules.
- Extracts tickers, company names, form types, time windows, fiscal periods,
  SEC item numbers, document roles, EX-99 types, and speaker hints where
  detectable.
- Plans temporal, cross-company, cross-source, speaker-specific, market-context,
  or single-document retrieval subqueries.
- Runs Phase 2 dense retrieval for each subquery and merges results.
- Hydrates bounded nearby same-document chunks as parent context.
- Validates citation labels only against retrieved chunks.
- Prints routing, filters, subqueries, top-k results, coverage, and citation
  validation.

## Limitations And Deferred Work

- Routing is rule-based and intentionally conservative.
- Temporal subqueries preserve trace metadata but do not yet filter by fiscal
  quarter unless the local chunk metadata has that field.
- Parent context is hydrated from neighboring chunks, not parent-child
  embeddings.
- No answer synthesis, agents, frontend, Qdrant, sparse retrieval, reranking,
  paid transcripts, OpenAI, Anthropic, or production evals are included.
