# Financial RAG Phase 1 Smoke Pipeline

Phase 1 adds a one-ticker SEC ingestion-to-embedding smoke path. It is scoped to
local files and offline-verifiable pipeline pieces only: no FastAPI service,
frontend, answer synthesis, agents, hybrid retrieval, reranking, paid transcript
sources, or production evals.

## Command

```bash
.\venv\Scripts\python.exe -m scripts.financial_rag_phase1_smoke
```

The default ticker is `NVDA`. Use `--ticker MSFT` to run another ticker. The
script selects the latest exact-form `10-K`, latest exact-form `10-Q`, recent
exact-form `8-K` primary documents, and EX-99 exhibits discovered from recent
8-K filing index pages.

## Environment

- `SEC_USER_AGENT` is required and must be a real SEC fair-access contact string,
  such as `Your Name your.email@example.com`.
- `VOYAGE_API_KEY` is optional. When present and non-placeholder, the script uses
  Voyage AI embeddings and writes the local embedding cache. When absent, it
  skips embeddings with setup instructions.
- `ALPHA_VANTAGE_API_KEY` remains optional for Phase 0 transcript probing. The
  Phase 1 smoke script does not call Alpha Vantage.
- `OPENAI_API_KEY` may be present for later phases, but the Phase 1 smoke script
  does not read or call it.
- OpenAI and Anthropic are not called in Phase 1.

## Local Paths

All generated artifacts are under git-ignored paths:

- Raw SEC documents: `data/filings/raw/`
- Parsed text: `data/filings/parsed/`
- Chunks: `data/filings/chunks/`
- Embedding cache: `data/vector_cache/`

Each area writes a `manifest.jsonl` file with source metadata. Document and chunk
metadata preserves ticker, CIK, accession, form type, filing date, source URL,
local path, document role, and EX-99 exhibit type where applicable.

## Supported Behavior

- SEC CIK lookup from `company_tickers.json`.
- SEC company submissions fetch from `data.sec.gov`.
- SEC fair-access request spacing and user-agent headers.
- SEC Archives URL construction for primary documents and filing indexes.
- EX-99 discovery from filing index pages.
- Idempotent local writes for raw documents, parsed text, chunks, manifests, and
  embeddings.
- Minimal SEC HTML/text extraction.
- Deterministic character-window chunking with stable chunk IDs.
- Voyage embedding cache through the official SDK when configured.

## Deferred Work

The Phase 1 smoke path intentionally defers production SEC section parsing,
speaker-aware chunking, parent-child chunking, transcript vendor integrations,
FastAPI, frontend work, answer synthesis, OpenAI routing/synthesis/eval, hybrid
retrieval, reranking, Qdrant, and production evaluation datasets.
