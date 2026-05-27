# Financial RAG Phase 2 Retrieval Foundations

Phase 2 improves local retrieval quality without adding frontend, agents,
answer synthesis, Qdrant, production evals, OpenAI calls, Anthropic calls, or
paid transcript integrations.

## Command

```bash
.\venv\Scripts\python.exe scripts\financial_rag_phase2_retrieval_smoke.py
```

Defaults:

- Query: `What risks does NVIDIA describe?`
- Ticker filter: `NVDA`
- Top-k: `5`

Useful filters:

```bash
.\venv\Scripts\python.exe scripts\financial_rag_phase2_retrieval_smoke.py --form-type 10-K --item-number 1A
.\venv\Scripts\python.exe scripts\financial_rag_phase2_retrieval_smoke.py --document-role exhibit --exhibit-type CFO_COMMENTARY
```

## Prerequisites

- Run the Phase 1 smoke pipeline first so `data/filings/chunks/` exists.
- Configure `VOYAGE_API_KEY` so the smoke script can embed the query.
- Generate cached chunk embeddings under `data/vector_cache/` by rerunning Phase
  1 with `VOYAGE_API_KEY`.
- `SEC_USER_AGENT` is not used by the Phase 2 retrieval smoke because it does
  not refetch SEC data.

## Behavior

- Reads local chunk JSONL files and cached Voyage embedding JSON files only.
- Uses dense-only cosine retrieval.
- Supports filters for ticker, form type, accession, document role, exhibit
  type, and SEC item number.
- Prints top-k scores, citation labels, source URLs, metadata, and excerpts.
- Validates returned citation labels against the retrieval set and reports
  accepted/rejected counts.

## Parser And Chunker Notes

- 10-K and 10-Q documents use basic SEC Item boundary detection when headings
  are visible in extracted text.
- 8-K primary documents use basic Item boundary detection such as `Item 2.02`.
- EX-99 documents use speaker-turn chunks only when obvious speaker labels are
  present; otherwise they use heading/paragraph-aware chunks.
- EX-99 classification uses filename, SEC description, and optional text
  heuristics for press releases, CFO commentary, prepared remarks,
  slides/presentations, and generic EX-99 exhibits.

## Limitations And Deferred Work

- Existing Phase 1 chunk files may still have simple v1 metadata until the
  ingestion smoke is rerun on a clean local chunk cache.
- Retrieval is dense-only in this scoped Phase 2 implementation.
- No sparse retrieval, RRF, reranker, parent-child chunking, answer synthesis,
  streaming endpoint, Qdrant, Langfuse tracing, production eval dataset, or
  frontend work is included.
- The tiny offline eval helpers currently provide Recall@k and MRR scaffolding
  plus NVDA fixture questions; they are not a production benchmark.
