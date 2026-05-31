# Financial RAG Testing Expansion

This phase expands the filings/RAG test surface before dashboard integration.
It focuses on whether retrieval finds the right source sections, not merely
whether generated citations are syntactically valid.

## Commands

Offline retrieval eval:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_retrieval_eval.py
```

Dry-run answer eval:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_answer_eval.py
```

Opt-in live answer eval:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_answer_eval.py --use-voyage --live
```

Useful filters:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_retrieval_eval.py --tickers NVDA --max-cases 10
.\venv\Scripts\python.exe scripts\financial_rag_expanded_answer_eval.py --tickers NVDA --max-cases 5 --live
```

## Coverage

The expanded fixture set includes 50 retrieval cases and 16 answer-eval cases
across all 12 cached issuers (NVDA, AMD, MSFT, AAPL, JPM, XOM, INTC, GOOGL, META,
AMZN, BAC, GS) plus an unsupported-ticker control. EX-99 and CFO-commentary
coverage varies by issuer, and INTC chunks carry no item-number metadata (its
filing labels live only in a trailing cross-reference index), so item-filtered
INTC queries return empty while non-item topics resolve (see
[financial-rag-eval-baseline.md](financial-rag-eval-baseline.md)). Some cases
continue to surface coverage gaps until more SEC cache is ingested.

Topics include:

- 10-K Item 1A risks,
- 10-Q risk updates,
- 8-K/EX-99 CFO commentary,
- press releases,
- prepared remarks/slides where available,
- revenue, gross margin, data center, supply, export controls, and capital
  allocation,
- temporal, cross-source, and cross-company questions.

## Metrics

Retrieval eval reports:

- Recall@k and MRR when relevant chunk ids are known,
- section/source hit rate,
- citation metadata completeness,
- evidence-quality pass rate,
- failure counts for wrong section/source, safe-harbor-only evidence, empty
  results, duplicate chunks, missing metadata, and unsupported tickers.

Answer eval reports:

- pass rate,
- hallucinated citation count,
- uncited factual sentence count,
- weak/safe-harbor-only evidence count,
- insufficient-evidence handling count.

Both scripts write JSON and CSV reports under ignored `artifacts/rag_eval/`.

## Cost Controls

Answer eval defaults to dry-run mode and does not call OpenAI. Live mode requires
`--live`. Use `--max-cases` and `--tickers` to keep live runs small. Use
`--use-voyage` only when `VOYAGE_API_KEY` is configured and you want live Voyage
query embeddings for retrieval.

## Known Current Gaps

- The local cache covers NVDA, AMD, MSFT, AAPL, JPM, and XOM; EX-99 and
  CFO-commentary coverage is uneven by issuer (only NVDA has cached CFO
  commentary; JPM has no cached EX-99 exhibits).
- Some chunks have sparse SEC item metadata, so Item 1A constraints may fail even
  when nearby risk text is present.
- Risk queries can retrieve valid but weak safe-harbor-style evidence instead of
  ideal Item 1A risk-factor evidence; the retriever down-weights safe-harbor-only
  chunks for operating-risk queries to mitigate this.
- Additional non-cached tickers remain as offline fixture targets to track
  scale-out readiness before more SEC corpora are ingested.
