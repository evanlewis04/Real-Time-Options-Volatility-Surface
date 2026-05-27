# Financial RAG Retrieval Repair Guide

This guide covers the retrieval repair workflow before any dashboard
integration. It keeps SEC filings as the backbone, uses only ignored local data
paths for corpus artifacts, and leaves OpenAI answer calls opt-in.

## Prerequisites

- Use `.\venv\Scripts\python.exe`.
- Set `SEC_USER_AGENT` before fetching SEC data.
- Set `VOYAGE_API_KEY` only when generating or refreshing Voyage embeddings.
- Set `OPENAI_API_KEY` only for opt-in live answer evals.
- Do not commit `data/filings/*`, `data/vector_cache/*`, or
  `artifacts/rag_eval/*`.

## Repair And Reingestion

Cache-only chunk repair, with no SEC requests:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_retrieval_repair.py --tickers NVDA
```

Explicit SEC expansion for the current small corpus:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_retrieval_repair.py --tickers AMD,MSFT,AAPL,JPM,XOM --fetch-sec --recent-8k-limit 3
```

The repair command rebuilds parsed text and SEC-aware chunks from cached raw
filings. It overwrites parsed/chunk files for repaired documents and rebuilds
chunk metadata, but it does not refetch SEC unless `--fetch-sec` is supplied.
Voyage embeddings are generated only when `--embed` is supplied; otherwise
offline evals use the lexical fallback.

Refresh Voyage embeddings for current chunks and prune stale vector files after
rechunking:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_retrieval_repair.py --tickers NVDA,AMD,MSFT,AAPL,JPM,XOM --embed --prune-stale-embeddings
```

For resumable embedding refreshes, bound each run:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_retrieval_repair.py --tickers NVDA,AMD,MSFT,AAPL,JPM,XOM --embed --max-embed-chunks 200 --embed-batch-size 16
```

The summary report includes `current_chunk_count`,
`current_embedding_count`, `missing_current_embedding_count`, and
`stale_embedding_count`.

## Eval Commands

Expanded retrieval eval:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_retrieval_eval.py --top-k 5 --per-subquery-k 8
```

Dry-run answer eval:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_answer_eval.py --top-k 5 --per-subquery-k 8
```

Opt-in live OpenAI answer eval:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_answer_eval.py --use-voyage --live
```

Live mode is gated by retrieval quality unless `--skip-retrieval-gate` is
explicitly supplied.

To evaluate with live Voyage query embeddings over the local vector cache:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_retrieval_eval.py --use-voyage --top-k 5 --per-subquery-k 8
```

## Gold Labels

Gold labels live as human-reviewable topic selectors in
`src/financial_rag/evaluation/gold.py`. The eval resolves those selectors to
current local chunk IDs at runtime, so labels remain usable after chunk IDs
change during reingestion.

After a parser or chunker change:

1. Run `financial_rag_retrieval_repair.py` for affected tickers.
2. Run the expanded retrieval eval.
3. Inspect `artifacts/rag_eval/expanded_retrieval_eval.json`.
4. Update `GOLD_LABEL_SPECS` only when the selected chunk is no longer the best
   evidence for that topic.

## Interpreting Metrics

- `section_source_hit_rate`: whether retrieved evidence satisfies expected
  ticker, source, section, and required-keyword constraints.
- `evidence_quality_pass_rate`: citation and metadata completeness over
  retrieved evidence.
- `mean_recall_at_k`: case-level Gold Recall@k over resolved gold chunk IDs.
- `mrr`: reciprocal rank of the first resolved gold chunk.
- `safe_harbor_only`: risk queries where all evidence is safe-harbor boilerplate.
- `coverage`: per-ticker local corpus coverage, including EX-99 chunk count and
  whether press releases, CFO commentary, or prepared remarks are cached.

Current initial targets are NVDA section/source hit rate above 70%, evidence
quality above 80%, Gold Recall@5 above 60%, and safe-harbor-only risk failures
near zero.

## Remaining Gaps

- Some non-NVDA section/source cases still need better source constraints and
  company-specific query tuning.
- EX-99 coverage is uneven by issuer; coverage reports now expose this directly.
  In the current local cache, NVDA has CFO commentary, AAPL/AMD/MSFT/XOM have
  press-release EX-99 coverage but no CFO commentary, and JPM has no cached
  EX-99 exhibits.
- Dashboard integration remains deferred until retrieval quality is stable.
