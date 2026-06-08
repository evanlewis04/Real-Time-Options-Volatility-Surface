# Financial Filings Intelligence + Options Volatility

A financial intelligence platform with two connected halves:

- **Filings RAG** — answers analyst questions over SEC filings (10-K / 10-Q / 8-K /
  EX-99) with **validated inline citations**, built on idempotent SEC ingestion,
  section-aware chunking, hybrid retrieval, and honest coverage reporting.
- **Options volatility workstation** — computes and visualizes implied-volatility
  surfaces, skew, term structure, and expected moves from real option chains.

The two meet in a **unified brief**: ask a question and get a cited filing answer
beside an options-market context panel, with every value labeled by source and
provenance. The defining principle throughout is **data honesty** — live, delayed,
fallback, synthetic, retrieved, and model-derived values are never presented as
something they are not.

## Why it's interesting

- **Citation discipline** — every factual answer maps to a retrieved chunk; the
  validator rejects hallucinated citation labels and fails closed.
- **Messy real-world ingestion** — SEC EDGAR filings parsed into sections,
  exhibits, and speaker turns, with uneven EX-99 / CFO-commentary coverage
  surfaced honestly rather than hidden.
- **A real eval harness** — retrieval and answer quality are measured (Recall@k,
  MRR, NDCG@k, section/source hit rate, citation validity), not just asserted.
- **Provenance everywhere** — fitted surfaces, fallback chains, and retrieved
  evidence carry explicit "not a market observation" labels.
- **Measured, not bolted-on** — features that didn't earn their place (e.g. a
  reranker) ship as clearly-labeled opt-in infrastructure, not the default.

## Demo

The unified brief — a cited filing answer beside an options-market context panel,
each labeled as a distinct data source, with an opt-in citation-validated answer:

![Unified filing + market brief](docs/assets/brief-view.png)

The filings corpus is local-only (not committed). Build a small demo corpus once
— a free SEC fetch; **no paid keys required** (set `SEC_USER_AGENT` in `.env`;
Voyage embeddings are optional, lexical retrieval works without them):

```bash
.\venv\Scripts\python.exe scripts\financial_rag_retrieval_repair.py --tickers NVDA --fetch-sec
```

Then:

```bash
# 1) Unified brief: cited filing answer + options-market context (one screen)
.\venv\Scripts\python.exe -m streamlit run scripts\financial_rag_brief_view.py

# 1b) Headless version (writes a JSON brief, no UI)
.\venv\Scripts\python.exe scripts\financial_rag_brief_smoke.py

# 2) Options volatility workstation (runs offline on synthetic fallback data)
.\venv\Scripts\python.exe -m streamlit run app.py
```

Add `--embed` to the corpus command (with `VOYAGE_API_KEY`) for dense retrieval,
and `--answer` to the brief smoke (with `OPENAI_API_KEY`) for the opt-in
generated answer.

The volatility workstation shows data provenance, surface quality, and fit
diagnostics in one view:

![Dashboard overview](docs/assets/dashboard-overview.png)
![3D implied-volatility surface](docs/assets/dashboard-surface-3d.png)

## Eval results

Local retrieval/answer eval over a 12-ticker SEC corpus (~6,527 chunks). Numbers
are regenerated locally; see [docs/financial-rag-eval-baseline.md](docs/financial-rag-eval-baseline.md).

| Metric | Value |
| --- | --- |
| Eval cases / gold labels | 50 / 64 |
| Section/source hit rate | 0.980 |
| Gold Recall@5 | 0.682 |
| Gold MRR | 0.455 |
| Answer citation validity | 1.000 (0 hallucinated, 0 uncited) |

```bash
.\venv\Scripts\python.exe scripts\financial_rag_expanded_retrieval_eval.py --top-k 5 --per-subquery-k 8
.\venv\Scripts\python.exe scripts\financial_rag_expanded_answer_eval.py --top-k 5 --per-subquery-k 8
```

## Quick start

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt   # or requirements.lock for pinned installs
copy .env.example .env            # optional; needed only for live data / Voyage / OpenAI
```

The default workflows are cache-only and run offline. SEC ingestion (Voyage
embeddings optional) is opt-in:

```bash
.\venv\Scripts\python.exe scripts\financial_rag_retrieval_repair.py --tickers NVDA --fetch-sec --embed
```

## Key commands

| Command | What it does |
| --- | --- |
| `streamlit run scripts\financial_rag_brief_view.py` | Unified brief: cited filing evidence + market context. |
| `streamlit run app.py` | Options volatility workstation. |
| `python scripts\financial_rag_expanded_retrieval_eval.py` | Retrieval-quality eval (Recall@k, MRR, NDCG@k). |
| `python scripts\financial_rag_retrieval_repair.py --fetch-sec --embed` | Build/refresh the local SEC corpus and Voyage vectors. |
| `python scripts\financial_rag_api_server.py` | Optional local FastAPI adapter over the cache-only RAG service. |
| `python scripts\verify.py` | Lint, compile, full pytest suite, and dashboard healthcheck. |

## Architecture

```text
src/
|-- financial_rag/        # SEC ingestion, parsing, chunking, retrieval, query,
|   |                     #   synthesis (citations), evaluation, audit, API,
|   |                     #   differentiators, and the market-context integration
|-- data/                 # yfinance providers, normalized models, synthetic fallback
|-- pricing/ quant/ analysis/   # Black-Scholes, Greeks, IV solver, SVI, surface fitting
`-- dashboard/            # Streamlit volatility workstation
```

The data path is documented in [docs/architecture.md](docs/architecture.md), the
RAG-plus-market integration in
[docs/financial-rag-market-context.md](docs/financial-rag-market-context.md),
reviewer definitions in [docs/glossary.md](docs/glossary.md), and surface-fit
quality and provenance in [docs/surface_quality.md](docs/surface_quality.md).

## Honest limitations

- EX-99 / CFO-commentary coverage is uneven by issuer; coverage reports expose
  this rather than hiding it. Some tickers have primary filings only.
- INTC chunks lack item-number metadata (its 10-K labels live only in a trailing
  index), so item-filtered INTC queries return empty; non-item topics resolve.
- A rerank stage exists but ships **opt-in, off by default**: it does not beat the
  domain-tuned first stage on the current eval, partly because the gold labels are
  resolved by that same scorer. The honest write-up is in the eval-baseline note.
- yfinance market data may be delayed, incomplete, or rate-limited; the UI labels
  live / delayed / fallback / synthetic states explicitly.

## Testing

```bash
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe scripts\verify.py
```

Tests cover Black-Scholes pricing and Greeks, IV round trips, option-chain
cleaning, provider contracts, surface fitting and validation, SEC parsing and
chunking, retrieval and citation validation, eval metrics, API contracts, and the
unified brief.

## Disclaimer

For research and educational use. Not investment advice and not built for live
trading.
