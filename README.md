# Real-Time Options Volatility Surface

A Python toolkit for computing and visualizing implied-volatility surfaces for
US equity options. The dashboard now behaves as a compact quant workstation:
it shows data provenance, distinguishes live/delayed data from synthetic
fallbacks, builds surfaces from real `yfinance` option chains when available,
and exposes chain quality diagnostics beside the charts.

## Features

- Black-Scholes pricing and Greeks: delta, gamma, theta, vega, rho.
- Implied-volatility solver with Newton-Raphson, bisection, and Brent paths.
- Volatility surface construction across strikes and expiries.
- Streamlit workstation with Surface, Chain, Skew & Term, Risk, and Diagnostics tabs.
- Real yfinance option-chain normalization with source, cache, and rejection metadata.
- Explicit synthetic/fallback labeling when live or delayed data is unavailable.
- Realized return correlations from historical yfinance closes.
- Built-in caching to be friendlier to yfinance rate limits.

## Supported Tickers

Common defaults include `AAPL`, `MSFT`, `GOOGL`, `NVDA`, `TSLA`, `SPY`,
`QQQ`, `META`, `AMZN`, and `JPM`. The dashboard sidebar includes a larger
equity and ETF universe.

## Quick Start

```bash
git clone <this-repo>
cd Real-Time-Options-Volatility-Surface
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

For reproducible installs, use the pinned direct-dependency lock after creating
the virtual environment:

```bash
pip install -r requirements.lock
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

The `.env` file is optional for the default yfinance workflow. Paid API keys
are reserved for future provider integrations.

## Running

| Command | What it does |
| --- | --- |
| `streamlit run app.py` | Launches the interactive dashboard on port 8501. |
| `python main.py` | Runs a noninteractive smoke test and exits. |
| `python main.py --smoke-test` | Same as above, explicit CI-friendly smoke test. |
| `python main.py --interactive` | Opens the legacy interactive CLI prompt. |
| `python main.py test` | Runs the older full CLI system test, including data fetches. |
| `python scripts/verify.py` | Runs lint, compile, pytest, and the dashboard healthcheck. |
| `python scripts/verify.py --fix` | Applies Ruff formatting/fixes, then runs verification. |
| `python -m scripts.healthcheck` | Runs project import, pricing, connector, surface, and Streamlit checks. |
| `python diagnostic.py` | Sanity-checks pricing, IV, Greeks, and surface construction. |

## Dashboard State

The first screen exposes provenance before analysis. Live/delayed data uses the
market provider path; synthetic and fallback states are labeled in the header
and Diagnostics tab so generated data is never presented as live.

![Dashboard default AppTest state](docs/assets/dashboard-default-state.svg)

## Project Layout

```text
.
|-- app.py                  # Streamlit dashboard entry point
|-- main.py                 # CLI smoke test, monitoring, and legacy prompt
|-- dashboard_connector.py  # Data/provenance orchestration for the dashboard
|-- config.py               # Shared configuration
|-- requirements.lock       # Pinned direct dependency set
|-- docs/                   # Architecture notes and dashboard screenshots
|-- src/
|   |-- data/               # yfinance price/options providers and synthetic fallback
|   |-- pricing/            # Black-Scholes and implied-vol solver
|   |-- analysis/           # Volatility surface construction
|   |-- visualization/      # Plotly and matplotlib helpers
|   |-- portfolio/          # Portfolio analytics scaffolding
|   |-- realtime/           # Streaming/refresh scaffolding
|   |-- utils/              # Shared helpers
|   `-- config/             # Per-asset configuration
|-- tests/                  # pytest unit tests
|-- scripts/                # CLI utilities and healthcheck
`-- plots/                  # Generated surface images
```

## Testing

```bash
pytest tests/
python -m scripts.healthcheck
python scripts/verify.py
```

Tests cover Black-Scholes pricing, put-call parity, Greeks signs/bounds,
price-to-IV round trips, option-chain cleaning, provider contracts, and
deterministic Streamlit AppTest states.

## Architecture And Glossary

The data path is documented in [docs/architecture.md](docs/architecture.md):
providers -> normalized canonical models -> quant engine -> connector ->
dashboard, with offline tests attached to the normalized and rendered states.
Reviewer-facing definitions for IV, DTE, moneyness, skew, risk reversal,
butterfly, IV rank, SVI, and data provenance modes live in
[docs/glossary.md](docs/glossary.md).
Surface-fit quality and the distinction between raw quotes and fitted estimates
are covered in [docs/surface_quality.md](docs/surface_quality.md).

## Data Honesty

The dashboard intentionally labels each view as live/delayed, synthetic, or
fallback. yfinance data may be delayed, incomplete, rate-limited, or unavailable
for some symbols. When real option chains cannot be fetched or normalized, the
surface can fall back to a Black-Scholes-consistent synthetic chain, and the UI
shows that fallback reason in the data-quality row and Diagnostics tab.

## Disclaimer

This project is for research and educational use. It is not investment advice
and is not built for live trading.
