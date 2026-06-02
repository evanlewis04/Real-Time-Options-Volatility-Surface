# Financial RAG Eval Baseline

Frozen baseline for the local filings RAG retrieval and answer pipeline. Future
retrieval changes must beat or explain these numbers. Metrics are regenerated
locally; the underlying `artifacts/rag_eval/*` JSON/CSV are intentionally not
committed.

## Snapshot

- Date: 2026-05-31
- Branch: main
- Corpus: local cache only, 6 tickers, 48 filing documents, 2,922 chunks,
  2,922 Voyage embeddings (0 missing, 0 stale).
- Routing: deterministic (no LLM routing).

## Commands

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_retrieval_eval.py --top-k 5 --per-subquery-k 8
.\venv\Scripts\python.exe scripts\financial_rag_expanded_retrieval_eval.py --use-voyage --top-k 5 --per-subquery-k 8
.\venv\Scripts\python.exe scripts\financial_rag_expanded_answer_eval.py --top-k 5 --per-subquery-k 8
```

## Retrieval Eval (30 cases, 7 companies incl. 1 unsupported-ticker control)

| Metric | Offline (lexical) | Voyage |
| --- | --- | --- |
| Section/source hit rate | 0.933 | 0.933 |
| Metadata completeness | 0.967 | 0.967 |
| Evidence-quality pass rate | 0.967 | 0.967 |
| Gold Recall@5 | 0.625 | 0.625 |
| Gold MRR | 0.385 | 0.355 |
| Gold labels resolved | 29 | 29 |

Failure counts (both modes): `wrong_section_or_source` = 1, `unsupported_ticker` = 1.

- `unsupported-tsla-risk` is an intended negative control (TSLA not in corpus).
- `jpm-capital-return` is the one true retrieval miss — JPM has no cached EX-99
  exhibits, so capital-return commentary falls back to primary-filing wording.

## Answer Eval (16 cases, dry-run, no OpenAI calls)

| Metric | Value |
| --- | --- |
| Pass rate | 1.000 |
| Hallucinated citations | 0 |
| Uncited factual sentences | 0 |
| Weak/safe-harbor-only evidence cases | 0 |
| Evidence-quality pass rate | 1.000 |
| Gold Recall@5 | 0.714 |
| Retrieval errors | 0 |

## Corpus Coverage And Known Gaps

| Ticker | Chunks | EX-99 chunks | Press release | CFO commentary | Notable gap |
| --- | --- | --- | --- | --- | --- |
| NVDA | 413 | 121 | yes | yes | no prepared remarks |
| AMD | 435 | 63 | yes | no | no CFO commentary |
| MSFT | 430 | 48 | yes | no | no CFO commentary |
| XOM | 403 | 93 | yes | no | no CFO commentary |
| AAPL | 194 | 10 | yes | no | thin EX-99 coverage |
| JPM | 1,047 | 0 | no | no | no cached EX-99 narrative at all |

- EX-99 / CFO-commentary coverage is uneven by issuer; only NVDA has cached CFO
  commentary. JPM has primary filings only.
- Non-NVDA section/source tuning is the next planned work; the lone retrieval
  failure (`jpm-capital-return`) reflects this gap.
- These numbers are the floor for the upcoming non-NVDA retrieval tuning and
  gold-label expansion work.

## Update 2026-05-31: Retrieval Tuning, Gold Labels, And Corpus Expansion

The metrics above remain the frozen floor. The following reflect current state
after the capital-return retrieval tuning, gold-label expansion, the corpus
growth to 12 tickers, and the INTC chunking fix. Regenerate locally before
reuse; artifacts stay ignored.

Corpus now: 12 tickers, ~6,527 chunks, all Voyage-embedded (0 missing, 0 stale).

Retrieval eval (50 cases, 13 companies incl. 1 unsupported-ticker control):

| Metric | Offline (lexical) | Voyage |
| --- | --- | --- |
| Section/source hit rate | 0.980 | 0.980 |
| Evidence-quality pass rate | 0.980 | 0.980 |
| Gold Recall@5 | 0.682 | 0.659 |
| Gold MRR | 0.455 | 0.440 |
| Gold labels resolved | 64 | 64 |

Only `unsupported-tsla-risk` fails (intended control). Dry-run answer eval is
unchanged (1.000 pass, 0 hallucinated/uncited).

New-ticker coverage (added INTC, GOOGL, META, AMZN, BAC, GS):

| Ticker | Chunks | EX-99 chunks | Press release | CFO commentary | Notable gap |
| --- | --- | --- | --- | --- | --- |
| GS | 1,047 | 0 | no | no | no cached EX-99 narrative (like JPM) |
| BAC | 939 | 174 | yes | no | no CFO commentary |
| META | 534 | 18 | no | yes | has CFO commentary; no press release |
| INTC | 439 | 0 | no | no | no item-number metadata (see note) |
| AMZN | 331 | 51 | yes | no | no CFO commentary |
| GOOGL | 315 | 19 | yes | no | no CFO commentary |

- INTC chunking is fixed: its 10-K labels live only in a trailing cross-reference
  index, so the section parser now falls back to whole-document chunking instead
  of dropping the body (41 -> 439 chunks). The trade-off is that INTC chunks carry
  no item-number metadata, so item-filtered INTC queries (e.g. risk factors ->
  Item 1A) return empty; non-item topics (data center, manufacturing, competition)
  resolve and have eval cases.
- GS, like JPM, has no cached EX-99 narrative (investment-bank disclosure style).
- META is the only newly added ticker with cached CFO commentary.

## Reranker (opt-in, default off)

A swappable rerank stage exists (`src/financial_rag/retrieval/rerank.py`): a
deterministic, offline `LexicalRerankerV1` (BM25-lite) and an opt-in online
`VoyageReranker`, fused with the first stage via Reciprocal Rank Fusion. The eval
adds NDCG@k and a `--reranker none|lexical|voyage` flag.

The default is `none`: reranking does not beat the domain-tuned first stage on
this eval. Offline comparison (`--top-k 5 --per-subquery-k 8`):

| Config | Recall@5 | MRR | NDCG@5 | Hit rate |
| --- | --- | --- | --- | --- |
| none (baseline) | 0.682 | 0.455 | 0.481 | 0.980 |
| lexical, wide pool | 0.727 | 0.470 | 0.491 | 0.940 |
| lexical, protected-set | 0.682 | 0.418 | 0.453 | 0.980 |
| voyage, protected-set | 0.659 | 0.436 | 0.488 | 0.980 |

No config beats the baseline without a guardrail regression. Two reasons:

1. The first stage is already tuned to these cases (domain bonuses, safe-harbor
   suppression), so a generic reranker mostly adds noise.
2. The eval is partly circular: `gold.py` resolves each gold chunk with the
   first-stage `lexical_relevance_score`, so MRR/NDCG reward agreement with the
   first stage and penalize any reranker that reorders. A fair rerank measurement
   needs scorer-independent gold labels first.

The rerank stage ships as opt-in infrastructure (enable with `--reranker lexical`
or `reranker="voyage"`); it is wired in the protected-set configuration, which
reorders only the first-stage result set so source-hit and recall cannot regress
when enabled.
