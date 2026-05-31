# Financial RAG Next Steps Handoff

This document is a restart point for a new coding session, Codex run, or Claude
Code task. It summarizes the current project state, guardrails, local commands,
and the recommended implementation sequence after the retrieval-repair commit.

## Current Project State

Repository:

- Project: Real-Time Options Volatility Surface.
- Current direction: expand the existing volatility workstation into a financial
  volatility and filings intelligence platform.
- Existing market app remains the volatility dashboard and market-context layer.
- New filings/RAG work lives beside it under `src/financial_rag/`.

Most recent committed RAG milestone:

- Commit: `d350580 Add financial RAG retrieval repair platform`.
- Branch pushed: `main` to `origin/main`.
- The working tree was clean immediately after push.

The committed RAG work includes:

- SEC-backed local ingestion/parsing/chunking/retrieval scaffolding.
- SEC-aware chunk repair and multi-company corpus expansion scripts.
- Deterministic query planning rules.
- Local dense retrieval with lexical fusion and Voyage vector-cache support.
- Gold-label based retrieval evals.
- Dry-run and opt-in OpenAI answer evals.
- Evidence-quality checks, citation validation, coverage reporting, and local
  API/workbench smoke paths.
- Documentation for phases, testing expansion, OpenAI testing, retrieval repair,
  and API/demo behavior.

The project should not yet integrate filings RAG into the main volatility
dashboard. Retrieval quality should be stabilized first.

## Important Guardrails

Use the local virtual environment explicitly:

```powershell
.\venv\Scripts\python.exe
```

Data-source and provider rules:

- SEC EDGAR remains the filing backbone.
- Use only free or free-tier data sources unless the user explicitly changes the
  project policy.
- Voyage remains the embeddings provider through `VOYAGE_API_KEY`.
- OpenAI can be used only for opt-in live answer tests through
  `OPENAI_API_KEY`.
- Do not call Anthropic.
- Do not add paid transcript APIs.
- Do not add OpenAI routing yet; query planning should remain deterministic.

Commit hygiene:

- Do not commit `.env`.
- Do not commit SEC raw filings, parsed filings, chunks, embeddings, vector
  cache contents, generated eval reports, dashboard screenshots, logs, or other
  generated artifacts.
- The following local artifact roots are intentionally ignored:
  - `data/filings/raw/*`
  - `data/filings/parsed/*`
  - `data/filings/chunks/*`
  - `data/vector_cache/*`
  - `artifacts/rag_eval/*`
- The `.gitkeep` placeholders under ignored data folders are intentionally
  tracked.

Compatibility rules:

- Keep the volatility dashboard commands working.
- Keep Phase 1 through Phase 7 financial RAG commands working.
- Preserve provenance and citation metadata in new code.
- Default scripts must not refetch SEC unless the command explicitly includes a
  fetch flag such as `--fetch-sec`.

## Key Files And Guides

Read these first in a new session:

- `README.md`
- `docs/financial-rag-platform-plan.md`
- `docs/financial-rag-retrieval-repair.md`
- `docs/financial-rag-testing-expansion.md`
- `docs/financial-rag-openai-testing.md`
- `docs/financial-rag-phase7-api-guide.md`
- `docs/financial-rag-phase6-demo-guide.md`

Core source areas:

- `src/financial_rag/ingestion/`: SEC client and filing discovery.
- `src/financial_rag/parsing/`: SEC text, item, and exhibit parsing.
- `src/financial_rag/chunking/`: simple and SEC-aware chunking strategies.
- `src/financial_rag/retrieval/`: local dense/vector retrieval.
- `src/financial_rag/query/`: deterministic planning, routing, parent context,
  and local pipeline.
- `src/financial_rag/evaluation/`: expanded eval cases, gold label selectors,
  metrics, and reporting.
- `src/financial_rag/audit/`: readiness and evidence-quality checks.
- `src/financial_rag/synthesis/`: citation validation and OpenAI response path.
- `scripts/financial_rag_retrieval_repair.py`: cache rebuild, optional SEC
  fetch, optional Voyage embedding refresh.
- `scripts/financial_rag_expanded_retrieval_eval.py`: retrieval-quality eval.
- `scripts/financial_rag_expanded_answer_eval.py`: dry-run or opt-in live
  answer eval.

## Current Local Corpus And Metrics

The last completed retrieval-repair run produced a local cache with:

- Tickers: `NVDA`, `AMD`, `MSFT`, `AAPL`, `JPM`, `XOM`.
- Chunk count: `2922`.
- Voyage embeddings: `2922`.
- Missing current embeddings: `0`.
- Stale embeddings: `0`.

Last known retrieval metrics after repair:

- Offline retrieval section/source hit rate: `0.9333`.
- Offline evidence-quality pass rate: `0.9667`.
- Offline Gold Recall@5: `0.625`.
- Offline MRR: `0.3854`.
- Voyage retrieval section/source hit rate: `0.9333`.
- Voyage evidence-quality pass rate: `0.9667`.
- Voyage Gold Recall@5: `0.625`.
- Voyage MRR: `0.3549`.
- Dry-run answer eval pass rate: `1.0`.
- Hallucinated citations: `0`.
- Uncited factual sentences: `0`.

These metrics are local state, not committed artifacts. Regenerate them before
using them in README screenshots, project summaries, or PR descriptions.

Known retrieval/corpus gaps:

- Non-NVDA section/source hit rate still needs tuning. Failures are mostly
  source-keyword constraints and company-specific phrasing.
- EX-99 coverage varies by company.
- CFO commentary is not uniformly available:
  - NVDA has CFO commentary in the current cache.
  - AAPL, AMD, MSFT, and XOM have press-release style EX-99 coverage but no
    uniform CFO commentary.
  - JPM has no cached EX-99 exhibits in the current local cache.
- Dashboard integration remains deferred until retrieval quality is stable.

## Verification Baseline

Before the retrieval-repair commit, the following commands passed:

```powershell
.\venv\Scripts\python.exe -m ruff check scripts src tests
.\venv\Scripts\python.exe -m pytest tests\financial_rag -q
.\venv\Scripts\python.exe -m scripts.healthcheck
.\venv\Scripts\python.exe scripts\verify.py
```

Observed results:

- Ruff: all checks passed.
- Financial RAG tests: `75 passed`.
- Healthcheck: all checks passed.
- Full verify: `303 passed`.

Use at least the first three commands before handoff for any source change.
Run `scripts\verify.py` before a major commit or before pushing.

## Recommended Next Steps

The next project phase should be:

1. Freeze a reproducible eval baseline.
2. Tune non-NVDA retrieval.
3. Expand the corpus deliberately.
4. Promote the local API shape.
5. Improve a separate filings workbench.
6. Only then connect RAG evidence to volatility market context.

Do not start with dashboard integration. The platform will be stronger if the
retrieval and citation layer is demonstrably good before it gets a polished UI.

## Step 1: Freeze A Baseline

Goal:

- Regenerate retrieval and answer eval artifacts locally.
- Record a current eval snapshot in a short docs note or README table.
- Establish numbers that future retrieval changes must beat or explain.

Commands:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_retrieval_eval.py --use-voyage --top-k 5 --per-subquery-k 8
.\venv\Scripts\python.exe scripts\financial_rag_expanded_answer_eval.py --top-k 5 --per-subquery-k 8
```

Optional live answer smoke, only if `OPENAI_API_KEY` is configured:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_answer_eval.py --use-voyage --live
```

Expected output location:

- `artifacts/rag_eval/expanded_retrieval_eval.json`
- `artifacts/rag_eval/expanded_retrieval_eval.csv`
- `artifacts/rag_eval/expanded_answer_eval.json`
- `artifacts/rag_eval/expanded_answer_eval.csv`

Do not commit those artifacts.

Recommended committed output:

- Add a small doc section with the date, command, corpus coverage, and key
  metrics.
- If README is updated, keep it concise and link to a deeper eval note.

Implementation prompt:

```text
Read docs/financial-rag-next-steps-handoff.md and the retrieval repair guide.
Regenerate the expanded retrieval and dry-run answer eval baselines. Do not
commit artifacts under artifacts/rag_eval. Add a concise committed eval-baseline
note that records commands, corpus coverage, key metrics, and known gaps. Keep
the volatility dashboard working. Run ruff, tests/financial_rag, and
scripts.healthcheck.
```

## Step 2: Tune Non-NVDA Retrieval

Goal:

- Improve section/source hit rate and Gold Recall@5 for non-NVDA cases.
- Reduce failures caused by source-keyword constraints and company-specific
  wording.
- Keep safe-harbor-only risk failures near zero.

Likely code areas:

- `src/financial_rag/evaluation/expanded.py`
- `src/financial_rag/evaluation/gold.py`
- `src/financial_rag/query/planning.py`
- `src/financial_rag/query/router.py`
- `src/financial_rag/query/pipeline.py`
- `src/financial_rag/retrieval/local_dense.py`

Suggested tuning work:

- Add company-specific aliases for expected source terms.
- Expand keyword families for:
  - AI infrastructure and data center demand.
  - Cloud and intelligent cloud language.
  - Supply, export controls, China restrictions, capacity constraints.
  - Consumer/device phrasing for AAPL.
  - Energy upstream/downstream/commodity phrasing for XOM.
  - Banking, card services, capital, credit, and markets language for JPM.
- Improve source constraints so valid parent-context evidence can satisfy a
  source/section hit when the precise child chunk uses nearby wording.
- Keep deterministic query planning rules:
  - `Item 1A` or `risk factors`: force `10-K`/`10-Q` and `item_number=1A`
    where possible.
  - `CFO commentary`: prefer `EX-99` and CFO filters.
  - `press release`: prefer `EX-99` and press-release filters.
- Avoid overfitting by adding tests for both positive and negative cases.

Metrics to watch:

- `section_source_hit_rate`
- `evidence_quality_pass_rate`
- `mean_recall_at_k`
- `mrr`
- `safe_harbor_only`
- per-ticker coverage results

Implementation prompt:

```text
Use docs/financial-rag-next-steps-handoff.md. Implement one retrieval tuning
pass focused on non-NVDA section/source failures. Keep routing deterministic and
do not add OpenAI routing. Update tests for source constraints, company-specific
aliases, safe-harbor suppression, and gold-label resolution. Run expanded
retrieval eval before and after, summarize metric deltas, and do not commit eval
artifacts.
```

## Step 3: Expand Gold Labels

Goal:

- Move from a small initial gold-label set toward at least 50 labeled cases.
- Make Recall@5, MRR, and per-topic retrieval regressions more meaningful.

Main file:

- `src/financial_rag/evaluation/gold.py`

Current design:

- Gold labels are not hardcoded chunk IDs.
- They are human-reviewable topic selectors that resolve to current local chunk
  IDs at eval runtime.
- This makes labels more resilient after reingestion or rechunking.

Suggested topics:

- NVDA:
  - Item 1A.
  - supply.
  - export controls.
  - data center.
  - CFO commentary.
  - press releases.
- AMD:
  - AI/data center.
  - client/gaming.
  - supply or customer concentration.
  - press release.
- MSFT:
  - cloud/AI infrastructure.
  - capex.
  - risk factors.
  - press release.
- AAPL:
  - services.
  - iPhone/device demand.
  - China/geographic risk.
  - press release.
- JPM:
  - credit.
  - capital.
  - market risk.
  - consumer/community banking.
- XOM:
  - commodity prices.
  - upstream/downstream.
  - capex.
  - risk factors.
  - press release.

Documentation to update:

- `docs/financial-rag-retrieval-repair.md`, if the label maintenance workflow
  changes.
- A new or existing eval baseline note, if the label count changes.

Implementation prompt:

```text
Expand src/financial_rag/evaluation/gold.py toward 50 manually reviewed
selector-based labels across the currently cached six tickers. Keep labels
resilient to rechunking. Add or update offline tests showing that labels resolve
to current chunk IDs and that Recall@5/MRR are meaningful. Run the expanded
retrieval eval and summarize label count and metrics.
```

## Step 4: Expand The Corpus Deliberately

Goal:

- Grow from 6 cached tickers to roughly 10-12 before attempting the full
  20-company universe.
- Improve cross-company and sector comparison demos.

Recommended next tickers:

- `INTC`
- `GOOGL`
- `META`
- `AMZN`
- `BAC`
- `GS`

Why this batch:

- Adds semiconductor peer coverage beyond NVDA/AMD.
- Adds large platform/cloud companies for AI infrastructure comparisons.
- Adds finance peers for JPM comparison.
- Stays small enough to debug corpus quality manually.

Explicit SEC fetch command:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_retrieval_repair.py --tickers INTC,GOOGL,META,AMZN,BAC,GS --fetch-sec --recent-8k-limit 3
```

Embedding refresh command:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_retrieval_repair.py --tickers NVDA,AMD,MSFT,AAPL,JPM,XOM,INTC,GOOGL,META,AMZN,BAC,GS --embed --prune-stale-embeddings
```

For resumable embedding refreshes:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_retrieval_repair.py --tickers NVDA,AMD,MSFT,AAPL,JPM,XOM,INTC,GOOGL,META,AMZN,BAC,GS --embed --max-embed-chunks 200 --embed-batch-size 16
```

After expansion:

- Run coverage.
- Run expanded retrieval eval.
- Add or update gold labels for new tickers.
- Document EX-99 and CFO commentary coverage honestly.

Implementation prompt:

```text
Expand the local SEC corpus to INTC, GOOGL, META, AMZN, BAC, and GS using
--fetch-sec explicitly. Keep raw/parsed/chunks/vectors ignored. Refresh Voyage
embeddings with --embed and --prune-stale-embeddings. Update coverage reporting
or docs with which companies have press releases, CFO commentary, prepared
remarks, or no useful EX-99 narrative. Do not commit local corpus artifacts.
```

## Step 5: Promote The API Boundary

Goal:

- Turn the current local API adapter into a clearer backend contract before UI
  work.
- Keep it cache-only and evidence-first.

Existing guide:

- `docs/financial-rag-phase7-api-guide.md`

Current endpoints described:

- `GET /health`
- `GET /coverage?ticker=NVDA`
- `POST /query`
- `GET /chunks/{chunk_id}`
- `GET /differentiators/{ticker}`

Recommended next work:

- Decide whether to create a formal `backend/` FastAPI app now or harden the
  existing script path first.
- If creating `backend/`, keep the first version thin and local:
  - Import `src/financial_rag.api.local_service`.
  - Expose retrieval and coverage only.
  - Do not add database services yet.
  - Do not require Qdrant yet.
- Add contract tests for response shape, error codes, and citation metadata.
- Keep FastAPI optional only if preserving the current lightweight script path
  is more important than API polish.

Implementation prompt:

```text
Read docs/financial-rag-phase7-api-guide.md. Promote the local RAG API boundary
without adding databases, Qdrant, OpenAI routing, agents, or dashboard
integration. Add or harden endpoints for health, coverage, query evidence,
chunks, and differentiators. Add contract tests for success and structured
errors. Keep cache-only behavior by default.
```

## Step 6: Improve A Separate Filings Workbench

Goal:

- Give reviewers a useful evidence-first RAG interface before merging it into
  the volatility dashboard.

Current script:

- `scripts/financial_rag_phase4_workbench.py`

Recommended workbench improvements:

- Query input.
- Ticker selector.
- Retrieved evidence list.
- Citation metadata table.
- Source URL/accession/filing date visibility.
- Coverage/readiness warnings.
- Evidence-quality panel.
- Optional live-answer button gated behind retrieval quality and
  `OPENAI_API_KEY`.

Do not:

- Merge into the main volatility dashboard yet.
- Hide missing EX-99 or CFO coverage.
- Present generated answers without retrieved evidence and citation validation.

Implementation prompt:

```text
Improve the separate filings evidence workbench. Keep it outside the main
volatility dashboard. Show query results, citation metadata, coverage warnings,
and evidence-quality checks. Add an optional live answer path only if retrieval
passes the configured quality gate. Add focused tests or smoke checks and keep
existing dashboard tests passing.
```

## Step 7: Connect Market Context After RAG Stabilizes

Goal:

- Use the existing volatility engine as market context for cited disclosure
  answers.

Do this only after:

- Retrieval metrics are stable.
- Gold Recall@5 is above the initial target.
- Safe-harbor-only risk failures are near zero.
- Evidence-quality pass rate remains above target.
- The filings workbench/API can expose limitations clearly.

Target workflow:

```text
NVIDIA says data center demand accelerated. What did IV/skew do around the same
period?
```

Likely integration shape:

- RAG answers provide cited disclosure evidence.
- Market engine provides expected move, IV rank, skew, term structure, and
  provenance.
- UI clearly distinguishes filing evidence from market-data context.

Implementation prompt:

```text
Design a thin integration between financial RAG evidence and the existing
volatility market-context layer. Do not rewrite the dashboard. Add a cache-only
prototype that returns cited filing evidence plus market provenance for one
question/ticker. Keep data-source labels explicit and add tests for the combined
response shape.
```

## Suggested Milestone Order

Recommended next three commits:

1. `Document financial RAG eval baseline`
   - Regenerate local eval artifacts.
   - Commit only the written baseline note and any small doc updates.

2. `Tune non-NVDA RAG retrieval`
   - Improve source constraints and company-specific aliases.
   - Expand tests.
   - Commit code/tests/docs, not artifacts.

3. `Expand financial RAG gold labels`
   - Add more selector-based labels.
   - Improve metrics reliability.
   - Commit label specs/tests/docs.

Then:

4. `Expand local financial RAG corpus coverage`
5. `Harden local RAG API contract`
6. `Improve filings evidence workbench`
7. `Prototype market-context integration`

## Definition Of Done For The Next Sprint

A strong next sprint should end with:

- A committed eval-baseline note.
- Non-NVDA retrieval improvements with test coverage.
- At least 40-50 gold-label selectors.
- No generated filings/chunks/vectors/artifacts committed.
- `ruff`, focused financial RAG tests, healthcheck, and preferably full verify
  passing.
- Clear docs that say dashboard integration is still intentionally deferred.

## Standard Verification Commands

Run before handoff:

```powershell
.\venv\Scripts\python.exe -m ruff check scripts src tests
.\venv\Scripts\python.exe -m pytest tests\financial_rag -q
.\venv\Scripts\python.exe -m scripts.healthcheck
```

Run before major commit or push:

```powershell
.\venv\Scripts\python.exe scripts\verify.py
```

If eval metrics are relevant to the change:

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_expanded_retrieval_eval.py --use-voyage --top-k 5 --per-subquery-k 8
.\venv\Scripts\python.exe scripts\financial_rag_expanded_answer_eval.py --top-k 5 --per-subquery-k 8
```

## Notes For A New Coding Agent

- Start by checking `git status --short --untracked-files=all`.
- Assume ignored local corpus files may exist and are intentionally not tracked.
- Do not delete `data/filings/*` or `data/vector_cache/*` unless the user
  explicitly asks to clear local data.
- Use `rg` for search.
- Prefer focused code changes plus tests over broad refactors.
- Preserve the current volatility dashboard behavior.
- Keep generated reports under ignored artifact paths.
- Summarize exact commands and results in the final handoff.
