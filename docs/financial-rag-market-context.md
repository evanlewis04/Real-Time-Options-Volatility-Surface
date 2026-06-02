# Financial RAG Market-Context Integration

A thin, cache-only prototype that connects cited filing evidence (from the local
RAG pipeline) with options-market context (from the existing volatility engine).
It answers questions like:

> NVIDIA says data center demand accelerated. What did IV and skew do around the
> same period?

The two data sources are kept explicitly separate. Filing evidence is management
disclosure with validated citations; market context is options-market-implied
data with its own provenance. They are never merged into a single claim.

## Commands

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_market_context_smoke.py
.\venv\Scripts\python.exe scripts\financial_rag_brief_smoke.py
.\venv\Scripts\python.exe -m streamlit run scripts\financial_rag_brief_view.py
```

The first writes the evidence + market-context brief. The second writes the
unified brief (evidence + optional gated answer + market context). The third
launches the unified brief view: one screen pairing cited filing evidence with a
market-context panel, with an opt-in gated answer.

All are cache-only by default: they use a deterministic offline market snapshot
labeled `Fallback`. Pass `--live-market` to source metrics from the volatility
engine, `--use-voyage` for live Voyage query embeddings, and `--answer` (smoke)
to attempt the gated OpenAI answer. Briefs are written under ignored
`artifacts/rag_eval/`.

## Unified brief

`build_unified_brief` (in `src/financial_rag/integration/brief.py`) composes the
market-evidence combiner with the Step-6 gated answer path into one structure:

- `filing_evidence`: cited disclosure (source `sec_filings_disclosure`).
- `market_context`: market-implied panel (source `options_market_implied`).
- `answer`: an optional citation-validated OpenAI answer, present only when
  `run_answer` is set and the evidence/readiness gate allows it.
- `answer_gate`: the gate decision and the reasons for any block.
- `data_sources` and `notes`: explicit labels keeping disclosure, market data,
  and any generated answer distinct.

The volatility dashboard is unchanged; market context is attached only through
the injectable provider seam.

## Combined brief shape

- `filing_evidence`: source `sec_filings_disclosure`, retrieved evidence rows,
  and accepted/rejected citations from the RAG pipeline.
- `market_context`: source `options_market_implied`, with `status`,
  `source_mode` provenance, and market-implied `metrics` (e.g. expected move, IV
  rank, skew).
- `data_sources`: explicit labels distinguishing cited disclosure from
  market-implied data, each with its provenance.
- `notes`: reminders that filing evidence is not market data, and that market
  reaction must not be inferred from filings alone.

## Design

- The combiner (`build_market_evidence_brief`) is pure and decoupled: it takes a
  RAG query payload and a `MarketContext`, so the integration package does not
  depend on the dashboard or quant stack.
- The volatility engine plugs in through an injectable provider. Use
  `market_provider_from_metrics(snapshot)` to adapt any precomputed market
  snapshot, or `volatility_market_provider` to lazily call the existing
  dashboard connector. Provider failures degrade to a labeled, unavailable
  market context rather than failing the brief.

## Limitations And Deferred Work

- This is an evidence-first prototype, not a dashboard merge; the main
  volatility workstation is unchanged.
- Live market metrics depend on the volatility engine's data mode and may be
  delayed, fallback, or synthetic; provenance is always labeled.
- Answer synthesis over the combined brief remains the workbench's opt-in,
  gated OpenAI path; this prototype returns evidence plus market provenance.
