# Financial RAG Market-Context Integration

A thin, cache-only prototype that connects cited filing evidence (from the local
RAG pipeline) with options-market context (from the existing volatility engine).
It answers questions like:

> NVIDIA says data center demand accelerated. What did IV and skew do around the
> same period?

The two data sources are kept explicitly separate. Filing evidence is management
disclosure with validated citations; market context is options-market-implied
data with its own provenance. They are never merged into a single claim.

## Command

```powershell
.\venv\Scripts\python.exe scripts\financial_rag_market_context_smoke.py
```

Cache-only by default: it uses a deterministic offline market snapshot labeled
`Fallback`. Pass `--live-market` to source metrics from the volatility engine
instead, and `--use-voyage` for live Voyage query embeddings. The brief is
written to ignored `artifacts/rag_eval/market_context_brief.json`.

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
