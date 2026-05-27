# Financial RAG Phase 4 Local Workbench

Phase 4 adds a local filings intelligence workbench over the Phase 3 query
pipeline. It is evidence-first: no answer synthesis, agents, Qdrant, paid
providers, OpenAI calls, Anthropic calls, or SEC refetch.

## Commands

```bash
.\venv\Scripts\python.exe scripts\financial_rag_phase4_workbench_smoke.py
.\venv\Scripts\python.exe -m streamlit run scripts\financial_rag_phase4_workbench.py
.\venv\Scripts\python.exe scripts\financial_rag_phase4_eval_report.py
```

## Prerequisites

- Local chunks under `data/filings/chunks/`.
- Cached vectors under `data/vector_cache/`.
- `VOYAGE_API_KEY` improves live workbench query retrieval, but the smoke script
  uses an offline constant query embedder.

## Behavior

- API-shaped local service exposes health, coverage, query, and chunk lookup
  methods, with an optional FastAPI adapter when FastAPI is installed.
- Streamlit workbench shows question input, ticker, top-k, routing, filters,
  subqueries, evidence rows, citation labels, source URLs, parent context,
  accepted/rejected citations, and coverage gaps.
- Eval report writes only to ignored `artifacts/rag_eval/`.

## Limitations And Deferred Work

- The workbench retrieves evidence; it does not synthesize answers.
- FastAPI is optional rather than a hard dependency.
- Eval fixtures are mostly unlabeled and are marked as such.
- Frontend frameworks, agents, Qdrant, sparse retrieval, reranking, paid
  transcript APIs, OpenAI, and Anthropic remain deferred.
