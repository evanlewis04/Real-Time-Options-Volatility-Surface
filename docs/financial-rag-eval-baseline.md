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

## Update 2026-07-31: Re-run On A Rolled-Forward Corpus

Regenerated locally after ~8 weeks by re-fetching all 12 tickers from SEC EDGAR
(`scripts/financial_rag_retrieval_repair.py --fetch-sec`). This pass is
**offline/lexical only** — the corpus was not re-embedded with Voyage (staying
inside free-tier limits), so only the offline column is refreshed here; dense
retrieval still works via `--embed` + `--use-voyage`. Environment: Python 3.13,
pandas 3.0 / numpy 2.5 / streamlit 1.60 (current majors). Full `scripts/verify.py`
is green (lint, compile, 315 tests, dashboard healthcheck).

Corpus now: 12 tickers, 76 documents, 6,259 chunks (vs the May snapshot's 6,527) —
the SEC filings rolled forward, so the underlying documents are not identical to
the May run.

Retrieval eval (50 cases, 13 companies incl. 1 unsupported-ticker control),
offline lexical:

| Metric | 2026-05-31 | 2026-07-31 |
| --- | --- | --- |
| Section/source hit rate | 0.980 | 0.860 |
| Metadata completeness rate | — | 0.900 |
| Evidence-quality pass rate | 0.980 | 0.900 |
| Gold Recall@5 | 0.682 | 0.667 |
| Gold MRR | 0.455 | 0.421 |
| Gold NDCG@5 | 0.481 | 0.436 |
| Gold labels resolved | 64 | 58 |

Failure counts: `empty_retrieval_results` = 4, `wrong_section_or_source` = 2,
`unsupported_ticker` = 1 (the intended TSLA control).

Answer eval (16 cases, dry-run, no OpenAI):

| Metric | 2026-05-31 | 2026-07-31 |
| --- | --- | --- |
| Pass rate | 1.000 | 0.812 |
| Hallucinated citations | 0 | 0 |
| Uncited factual sentences | 0 | 0 |
| Evidence-quality pass rate | 1.000 | 1.000 |
| Gold Recall@5 | 0.714 | 0.727 |
| Retrieval errors | 0 | 3 |

The deltas are corpus drift, not a code regression — parsing/chunking is
unchanged and fully tested, and **citation discipline holds exactly (0
hallucinated, 0 uncited)**. The four empty-retrieval cases are all coverage gaps
in the freshly-fetched filing window, consistent with the documented uneven-EX-99
limitation:

- `nvda-cfo-revenue`, `nvda-press-release-gross-margin` — NVDA's current 8-K
  window this cycle carries no chunked EX-99 press-release / CFO-commentary
  exhibit (in May it had 121 EX-99 chunks), so the two gold cases that target that
  exhibit find nothing. Other issuers' EX-99 exhibits (e.g. AAPL, AMD) still parse
  and chunk normally, so this is a missing-source gap, not a parser break.
- `xom-item1a-commodity-risk`, `xom-energy-transition` — XOM's refreshed corpus is
  thin (94 chunks) and the specific Item 1A / energy-transition sections those
  gold cases pin to are not present in the current filings.

Recovering these is a re-fetch/coverage question (pull the specific EX-99 exhibits
and XOM sections), not a retrieval-methodology change — left as-is here per the
"re-run only" scope.

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
