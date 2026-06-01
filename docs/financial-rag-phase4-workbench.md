# Financial RAG Phase 4 Local Workbench

Phase 4 adds a local filings intelligence workbench over the Phase 3 query
pipeline. It is evidence-first: retrieval, citations, coverage, and quality are
always shown before any answer. A generated answer is opt-in, OpenAI-only, and
gated behind retrieval quality plus `OPENAI_API_KEY`; no agents, Qdrant, paid
transcript providers, Anthropic calls, or SEC refetch.

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

- API-shaped local service exposes health, companies, coverage, documents,
  query, chunk lookup, differentiators, and market-context methods, with an
  optional FastAPI adapter when FastAPI is installed.
- Streamlit workbench shows a cached-ticker selector (from `/companies`),
  question input, top-k, routing, filters, subqueries, evidence rows, citation
  labels, source URLs, parent context, accepted/rejected citations, evidence
  quality, readiness, and coverage gaps.
- Grounded answers are opt-in: a dry-run prompt preview is always available at no
  cost, and a live OpenAI answer button is offered only when the per-query gate
  passes (retrieved evidence present, evidence quality `pass`, and OpenAI
  configured). Generated answers always show validated citations and drop any
  hallucinated citation labels; the gate explains every block.
- Eval report writes only to ignored `artifacts/rag_eval/`.

## Limitations And Deferred Work

- The workbench is evidence-first; answer synthesis is opt-in and never runs
  without retrieved evidence, a passing quality gate, and `OPENAI_API_KEY`.
- FastAPI is optional rather than a hard dependency.
- Eval fixtures are mostly unlabeled and are marked as such.
- Frontend frameworks, agents, Qdrant, sparse retrieval, reranking, paid
  transcript APIs, and Anthropic remain deferred.
