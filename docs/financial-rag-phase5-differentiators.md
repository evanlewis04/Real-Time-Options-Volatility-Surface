# Financial RAG Phase 5 Differentiators

Phase 5 adds local differentiators that make the filings workbench more useful
without LLMs, paid data, Qdrant, agents, or SEC refetch.

## Command

```bash
.\venv\Scripts\python.exe scripts\financial_rag_phase5_differentiators_report.py
```

## Prerequisites

- Local chunks under `data/filings/chunks/`.
- Optional local SEC companyfacts-style JSON under
  `data/companyfacts/{TICKER}.json`.
- Cached vectors are not required by the report command because it uses local
  differentiator helpers directly.

## Behavior

- Detects deterministic paragraph-level changes between sequential same-ticker
  10-K/10-Q item chunks when item metadata is available.
- Returns clear XBRL status from local companyfacts JSON, or `unavailable` when
  no local facts file exists.
- Scores uncertainty, risk, positive, and negative keyword signals with
  transparent term counts.
- Exposes a market-context hook that reports `unavailable` unless a provider is
  supplied.
- Writes reports only under ignored `artifacts/rag_eval/`.

## Limitations And Deferred Work

- Existing simple Phase 1 chunk caches may have sparse item metadata until
  regenerated with the SEC-aware chunker.
- XBRL helpers do not fetch SEC data.
- Language scoring is deterministic keyword counting, not ML sentiment.
- No answer synthesis, agents, Qdrant, paid transcripts, OpenAI, or Anthropic
  are included.
