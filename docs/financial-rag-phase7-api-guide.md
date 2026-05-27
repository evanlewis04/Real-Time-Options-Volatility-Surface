# Financial RAG Phase 7 Local API Guide

Phase 7 hardens the local API contract before answer synthesis. It keeps the
system cached, deterministic, and evidence-first: no SEC refetch, OpenAI,
Anthropic, paid transcript APIs, agents, Qdrant, or frontend framework work.

## Prerequisites

- Use `.\venv\Scripts\python.exe`.
- Keep chunks under `data/filings/chunks/`.
- Keep vectors under `data/vector_cache/`.
- Optional `data/companyfacts/{TICKER}.json` supports local XBRL scaffolding.
- FastAPI is optional. The smoke command works without it; the server command
  prints install guidance if FastAPI is missing.

## Commands

```bash
.\venv\Scripts\python.exe scripts\financial_rag_phase7_api_smoke.py
.\venv\Scripts\python.exe scripts\financial_rag_phase7_api_server.py
```

The smoke command writes `artifacts/rag_eval/phase7_api_smoke.json` by default.
The server command defaults to `127.0.0.1:8765` and uses cached local
chunks/vectors only.

## Endpoints

- `GET /health`: service status, chunk count, embedding count.
- `GET /coverage?ticker=NVDA`: local form/source coverage.
- `POST /query`: evidence retrieval payload for a question.
- `GET /chunks/{chunk_id}`: one local chunk by id.
- `GET /differentiators/{ticker}`: local Phase 5 differentiator payload.

Example query body:

```json
{
  "question": "What risks does NVIDIA describe?",
  "ticker": "NVDA",
  "top_k": 5,
  "per_subquery_k": 5
}
```

## Error Behavior

The local service raises structured errors with `code`, `message`, and
`details`. The optional FastAPI adapter returns those errors as JSON.

Covered cases include unsupported tickers, empty questions, invalid `top_k` or
`per_subquery_k`, missing chunks, missing cached embeddings, invalid chunk ids,
and empty retrieval results.

## Limitations And Deferred Work

- The API retrieves and validates evidence; it does not synthesize answers.
- FastAPI remains optional rather than a required dependency.
- Contract tests use fake local chunks/vectors and do not start a real server.
- Production evals, answer synthesis, frontend frameworks, Qdrant, agents, and
  paid transcript depth remain deferred.
