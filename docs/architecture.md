# Architecture

The dashboard is organized around a provenance-preserving data path. Providers
produce raw market inputs, normalization converts them into canonical models,
the quant layer enriches and fits those models, and the Streamlit workstation
renders both analytics and quality metadata.

```mermaid
flowchart LR
    Providers["Market Data Providers<br/>yfinance, demo provider, local snapshots"]
    Normalize["Normalization And Cleaning<br/>OptionQuote, MarketDataSnapshot, rejection buckets"]
    Quant["Quant Engine<br/>rates, dividends, IV, Greeks, SVI, surface quality"]
    State["Dashboard Connector<br/>cache, provenance, fallback routing, timing"]
    UI["Streamlit Workstation<br/>surface, chain, skew, risk, diagnostics"]
    Tests["Offline Verification<br/>fixtures, AppTest, healthcheck, CI"]

    Providers --> Normalize
    Normalize --> Quant
    Quant --> State
    State --> UI
    Normalize --> Tests
    State --> Tests
    UI --> Tests
```

## Provenance Contract

Every provider response carries source, mode, timestamp, cache age, fallback
reason, and row-count metadata. Synthetic and fallback paths remain explicit so
analytics can be rendered without pretending delayed or generated data is live.

## Offline Contract

Tests use deterministic fixtures and AppTest fallback modes. CI does not require
network data for the dashboard smoke path; live providers can fail and the app
must still render a labeled fallback state.
