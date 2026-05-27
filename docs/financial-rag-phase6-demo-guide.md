# Financial RAG Phase 6 Recruiter Demo Guide

Phase 6 makes the local filings platform easier to demo with reliability and
evidence-quality checks. It does not add agents, answer synthesis, Qdrant,
frontend frameworks, SEC refetch, OpenAI, Anthropic, or paid transcript APIs.

## Setup

- Use this workspace virtual environment: `.\venv\Scripts\python.exe`.
- Keep local chunks under `data/filings/chunks/`.
- Keep local vectors under `data/vector_cache/`.
- Optional XBRL scaffolding reads `data/companyfacts/{TICKER}.json`.
- `VOYAGE_API_KEY` is only needed for ingestion or live query embeddings; the
  Phase 6 demo workflow uses the offline constant query embedder.

## Commands

```bash
.\venv\Scripts\python.exe scripts\financial_rag_phase6_demo_workflow.py
.\venv\Scripts\python.exe -m streamlit run scripts\financial_rag_phase4_workbench.py
```

The demo command defaults to ticker `NVDA` and query
`What risks does NVIDIA describe?`. It writes JSON reports under ignored
`artifacts/rag_eval/`.

## Expected Output

The script prints a compact pass, warning, or fail status for:

- local cache readiness,
- query smoke,
- workbench helper smoke,
- differentiator report,
- tiny offline eval report.

It also prints prerequisites, artifact paths, and next actions. A warning is
acceptable for known local-data gaps such as missing companyfacts JSON or sparse
EX-99 categories.

## Workbench Behavior

The Streamlit filings workbench now includes:

- a Phase 6 readiness panel for missing embeddings, item metadata, EX-99
  coverage, companyfacts availability, and unsupported tickers,
- an evidence-quality panel after retrieval for citation-label validity, source
  URLs, ticker/date/accession metadata, parent context, and duplicate chunk IDs.

## Talking Points

- The platform is evidence-first: retrieved chunks, citation labels, source
  URLs, and metadata are visible before any future answer synthesis.
- Current gaps are surfaced as data-readiness issues rather than hidden.
- Phase 5 differentiators remain deterministic and local: filing changes,
  language signals, XBRL hooks, and market-context hooks.
- The volatility dashboard remains separate and unchanged.

## Limitations And Deferred Work

- Existing Phase 1 chunks may need regeneration to fill SEC item metadata.
- Companyfacts JSON is local-only and not fetched by Phase 6.
- Eval fixtures are tiny and mostly unlabeled.
- API hardening, answer synthesis, production evals, Qdrant, paid transcript
  depth, and frontend framework work remain deferred.
