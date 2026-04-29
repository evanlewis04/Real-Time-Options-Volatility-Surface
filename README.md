# Real-Time Options Volatility Surface

A Python toolkit for computing and visualizing implied-volatility surfaces for
US equity options. It pulls live option chains from Yahoo Finance via
`yfinance` (no paid API key required), prices options with a Black-Scholes
implementation, solves for implied volatility, and renders interactive 3D
surfaces in a Streamlit dashboard.

## Features

- Black-Scholes pricing and full Greeks (delta, gamma, theta, vega, rho)
- Implied-volatility solver (Newton-Raphson with bisection fallback)
- Volatility surface construction across strikes and expiries
- Interactive Streamlit dashboard with 3D Plotly surfaces
- Portfolio analytics (P&L, exposure, Greek aggregation)
- Built-in caching to be friendly to the `yfinance` rate limits

## Supported tickers (defaults)

`AAPL`, `MSFT`, `GOOGL`, `NVDA`, `TSLA`, `SPY`, `QQQ`, `META`, `AMZN`, `JPM`.
Override with the `DEFAULT_SYMBOLS` list in `config.py` or the symbol picker
in the dashboard.

## Quick start

```bash
git clone <this-repo>
cd Real-Time-Options-Volatility-Surface
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # optional, only if using paid APIs
```

## Running

| Command | What it does |
| --- | --- |
| `streamlit run app.py` | Launches the interactive dashboard on port 8501. |
| `python main.py` | Runs the CLI workflow: fetch data, build a surface, log results. |
| `python setup_dashboard.py` | One-time setup helper that verifies the environment and writes default plots to `plots/`. |
| `python diagnostic.py` | Sanity-checks data fetching and pricing for a single symbol — useful when debugging. |

`yfinance` is the primary data source and needs no API key. The
`ALPHA_VANTAGE_API_KEY`, `POLYGON_API_KEY`, and `IEX_CLOUD_API_KEY` slots in
`.env` are reserved for optional paid backends and are unused by default.

## Project layout

```
.
├── app.py                  # Streamlit dashboard entry point
├── main.py                 # CLI entry point
├── dashboard_connector.py  # Glue between data, pricing, and the dashboard
├── config.py               # Single source of configuration
├── src/
│   ├── data/               # yfinance fetchers and option-chain cleanup
│   ├── pricing/            # Black-Scholes + implied-vol solver
│   ├── analysis/           # Volatility surface construction
│   ├── visualization/      # Plotly / matplotlib surface plots
│   ├── portfolio/          # Position and P&L tracking
│   ├── realtime/           # Streaming/refresh loop
│   ├── utils/              # Shared helpers
│   └── config/             # Per-asset configuration
├── tests/                  # pytest unit tests
├── scripts/                # One-off CLI utilities
└── plots/                  # Generated surface images
```

## Testing

```bash
pytest tests/
```

Tests cover Black-Scholes pricing against textbook reference values, put-call
parity, Greeks signs/bounds, and a price → IV → price round-trip for the
implied-volatility solver.

## Disclaimer

This project is for research and educational use. It is not investment advice
and is not built for live trading. Option data from `yfinance` can be delayed
or incomplete.
