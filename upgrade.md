# Upgrade Plan: Real-Time Options Volatility Surface

This file is written for a future Codex implementation pass. It turns the current project review into an actionable backlog for making the dashboard look more professional, behave more honestly, and become much stronger as a quant project.

## Review Snapshot

Tested on 2026-05-02 in the project root.

Commands run:

```bash
venv\Scripts\python.exe -m pytest tests -q
venv\Scripts\python.exe -m compileall app.py dashboard_connector.py main.py src tests
venv\Scripts\python.exe diagnostic.py
venv\Scripts\python.exe -m streamlit run app.py --server.headless=true --server.port=8501 --browser.gatherUsageStats=false
```

Results:

- Unit tests passed: 30 passed, 4 expected warning cases.
- Python compile passed for app, main, dashboard connector, src, and tests.
- Streamlit server started and returned HTTP 200 at `http://localhost:8501`.
- Streamlit AppTest ran `app.py` with zero app exceptions.
- `diagnostic.py` exposed a Windows console encoding crash in one print path.
- Streamlit emitted repeated deprecation warnings for `use_container_width`; migrate to `width="stretch"` before removal after 2025-12-31.
- Browser visual testing through the in-app browser was blocked by local Node `22.18.0`; the browser harness requires `>=22.22.0`.

Current strengths:

- Clean initial BSM pricing implementation with Greeks.
- IV solver has Newton, bisection, and Brent paths.
- Synthetic options chain generator is Black-Scholes consistent.
- Surface construction has multiple fallbacks.
- Streamlit dashboard already has market table, Greeks, 3D surface, heatmap, smile, correlations, and performance sections.
- Existing pytest suite covers core pricing and IV round trips.

Critical current limitations:

- The UI markets itself as live/professional while several key dashboard sections are synthetic or random.
- Real options chains are not currently the core dashboard data source; synthetic chains dominate surface generation.
- Portfolio metrics, correlations, P&L, volume, bid-ask spread, contracts, and some fallback Greeks are random or placeholder-backed.
- The app is a single large `app.py`, which makes professional UI iteration and testing harder.
- The quant layer does not yet model dividends, early exercise, borrow, rates curves, no-arbitrage constraints, smoothing calibration, or surface diagnostics deeply enough for serious quant use.
- There is no data provenance panel, stale-data indicator, confidence score, or explicit demo/live mode boundary.

## Implementation Rules For Codex

- Preserve all existing tests and add new tests beside every meaningful quant/data change.
- Do not present synthetic or random values as live market data.
- Build in small vertical slices: data contract, calculation, UI rendering, tests.
- Prefer deterministic fixtures over randomness in tests and screenshots.
- Keep the dashboard dense, restrained, and work-focused. This is a quant tool, not a marketing page.
- Add clear labels for units: annualized vol, daily theta, DTE, moneyness, log-moneyness, dollars, basis points.
- Favor explicit data lineage: source, timestamp, delay, cache age, fallback reason, and confidence.
- Avoid broad refactors unless they unlock multiple items in this plan.
- After each implementation phase, run `pytest tests -q`, a Streamlit smoke test, and at least one app-level test.

## Phase 0: Honesty, Stability, And Baseline Polish

These should happen first because they prevent misleading output.

- [x] Add a persistent data mode badge: `Live`, `Delayed`, `Synthetic`, or `Fallback`.
  - Files: `app.py`, `dashboard_connector.py`.
  - Acceptance: every displayed section can reveal whether it is live, cached, synthetic, or fallback.

- [x] Replace random portfolio metrics with clearly labeled demo metrics or remove them from live mode.
  - Files: `dashboard_connector.py`, `app.py`.
  - Acceptance: no random portfolio VaR, Sharpe, drawdown, or P&L appears as if it were calculated from real positions.

- [x] Replace random correlation matrix with either real historical-return correlations or a disabled state.
  - Files: `dashboard_connector.py`, `src/portfolio/portfolio_analytics.py`, tests.
  - Acceptance: correlations are derived from price/return time series with a visible lookback window.

- [x] Replace random volume, bid-ask, and contracts in `get_current_data` with real source fields or mark them synthetic.
  - Files: `dashboard_connector.py`, `src/data/*`.
  - Acceptance: each market-table field has source metadata.

- [x] Fix Windows encoding failures in `diagnostic.py`.
  - Files: `diagnostic.py`.
  - Acceptance: `venv\Scripts\python.exe diagnostic.py` exits 0 on Windows cp1252 consoles.

- [x] Remove mojibake from `README.md` and app labels.
  - Files: `README.md`, `app.py`.
  - Acceptance: tree glyphs, bullets, arrows, and footer text render correctly or use plain ASCII.

- [x] Migrate Streamlit deprecated `use_container_width=True` calls to `width="stretch"`.
  - Files: `app.py`.
  - Acceptance: AppTest produces no `use_container_width` deprecation warnings.

- [x] Stop printing from quant modules during app runs.
  - Files: `src/analysis/vol_surface.py`.
  - Acceptance: surface construction uses logging, not `print`, and AppTest output stays clean.

- [x] Make `main.py` noninteractive by default or document it honestly.
  - Files: `main.py`, `README.md`.
  - Acceptance: `python main.py --smoke-test` runs and exits without waiting for stdin.

- [x] Add a single project health command.
  - Suggested command: `python -m scripts.healthcheck`.
  - Acceptance: healthcheck runs tests for imports, data provider status, Streamlit import, and sample surface construction.

## Phase 1: Professional Dashboard Redesign

Goal: make the first screen feel like a real quant workstation.

- [x] Split the app into layout modules instead of one monolithic `app.py`.
  - Suggested files:
    - `src/dashboard/app_shell.py`
    - `src/dashboard/components/metrics.py`
    - `src/dashboard/components/surface_panel.py`
    - `src/dashboard/components/market_table.py`
    - `src/dashboard/components/controls.py`
    - `src/dashboard/theme.py`
  - Acceptance: `app.py` becomes a small entry point that composes modules.

- [x] Add a true top command bar.
  - Include ticker search, underlying selector, DTE range, moneyness range, data source, refresh, and mode.
  - Acceptance: core workflow controls are visible without digging through the sidebar.

- [x] Convert the sidebar into a compact configuration rail.
  - Include model settings, data settings, filters, and display toggles.
  - Acceptance: no oversized sidebar text; controls are grouped and scannable.

- [x] Replace the large centered title with a compact terminal-style header.
  - Include product name, current symbol, spot, timestamp, market status, data mode.
  - Acceptance: more dashboard real estate is used for analysis instead of branding.

- [x] Introduce tabs for major workflows.
  - Suggested tabs: `Surface`, `Chain`, `Skew`, `Term Structure`, `Risk`, `Backtest`, `Diagnostics`.
  - Acceptance: each tab has one clear analytical purpose.

- [x] Add a dense KPI strip above the main surface.
  - Metrics: spot, change, ATM IV, 25 delta risk reversal, 25 delta butterfly, front/back term spread, IV rank, IV percentile, surface points, stale age.
  - Acceptance: KPIs are deterministic and sourced from current data.

- [x] Add professional theme tokens.
  - Use neutral background, subtle borders, limited accent colors, and consistent spacing.
  - Avoid a one-note blue/purple dashboard.
  - Acceptance: CSS variables or Streamlit theme config drive colors and spacing.

- [x] Replace generic Plotly defaults with a consistent chart template.
  - Files: `src/dashboard/theme.py`, all chart code.
  - Acceptance: all charts share fonts, margins, gridlines, color scales, hover labels, and legend style.

- [x] Use financial color semantics.
  - Green/red only where directional gain/loss or rich/cheap makes sense.
  - Use diverging colors for spreads and neutral sequential colors for vol levels.
  - Acceptance: no arbitrary rainbow charts for decision-critical views.

- [x] Add skeleton/loading states for slow data fetches.
  - Acceptance: UI never appears frozen during yfinance calls.

- [x] Add empty/error states with clear recovery actions.
  - Acceptance: no raw exception dumps in normal user flow.

- [x] Add compact help tooltips instead of long in-app explanatory text.
  - Acceptance: Greeks explanations move into concise tooltips or docs panel.

- [x] Add responsive layout checks.
  - Acceptance: app remains usable at 1440px, 1024px, and mobile-width browser sizes.

- [x] Add visual regression screenshots for the main dashboard.
  - Suggested: use Streamlit AppTest for structural checks and Playwright when available.
  - Acceptance: automated test verifies key sections render without exceptions.

- [x] Make all tables sortable, filterable, and exportable.
  - Suggested: use `st.dataframe` column config or AgGrid where valuable.
  - Acceptance: chain and market tables can be filtered by expiry, moneyness, type, liquidity, and IV.

- [x] Add column formatting.
  - Price: dollars, IV: percent, theta: dollars/day, rates: percent, spreads: bps or percent.
  - Acceptance: no ambiguous raw decimals in the UI.

- [x] Add a "last refresh" and "cache age" indicator beside every dataset.
  - Acceptance: a user can tell stale live data from fresh live data.

- [x] Replace the footer with a compact diagnostics/status row.
  - Acceptance: no marketing-style footer inside the analytical app.

## Phase 2: Real Data Pipeline

Goal: graduate from synthetic dashboard demo to real market-data analysis with transparent fallbacks.

- [x] Add a canonical `MarketDataSnapshot` data model.
  - Fields: symbol, spot, spot_timestamp, chain_timestamp, expirations, options, source, source_delay, cache_age, fallback_reason.
  - Acceptance: dashboard consumes this model instead of loose dictionaries.

- [x] Add a canonical `OptionQuote` model.
  - Fields: contract, type, strike, expiry, dte, bid, ask, mid, last, volume, open_interest, raw_iv, computed_iv, delta, gamma, theta, vega, rho, quote_timestamp.
  - Acceptance: calculations and UI use typed/validated quote fields.

- [x] Fetch real option chains from yfinance where available.
  - Files: `src/data/price_provider.py`, new `src/data/options_provider.py`.
  - Acceptance: selected symbol surfaces can be built from real option-chain bid/ask/mid data.

- [x] Add provider abstraction for multiple market data sources.
  - Suggested providers: yfinance, Polygon, Tradier, Databento, Cboe delayed files, local CSV fixture.
  - Acceptance: connector can swap providers without dashboard changes.

- [x] Add data-source capability matrix.
  - Acceptance: UI shows which fields are available from each provider.

- [x] Cache option chains by symbol and expiry.
  - Acceptance: repeated dashboard reruns do not hammer yfinance.

- [x] Persist snapshots locally.
  - Suggested: parquet files under `data/snapshots/`.
  - Acceptance: app can replay a previous snapshot offline.

- [x] Add a historical price loader.
  - Acceptance: returns, realized vol, IV rank, and correlations use real history.

- [x] Add rate-limit handling and backoff.
  - Acceptance: provider failures produce graceful stale-data fallback with reason.

- [x] Add market calendar support.
  - Suggested: `pandas_market_calendars`.
  - Acceptance: app knows market open, close, holidays, and delayed/weekend states.

- [x] Add risk-free rate source.
  - Suggested: Treasury curve/FRED or a configurable curve file.
  - Acceptance: IV calculations use expiry-specific rates, not one hard-coded 5 percent value.

- [x] Add dividend yield and discrete dividend support.
  - Acceptance: equity option IVs can incorporate dividends by symbol/expiry.

- [x] Add corporate action awareness.
  - Acceptance: splits/dividends can be surfaced in data warnings.

- [x] Add stale quote filtering.
  - Acceptance: old or zero bid/ask quotes are excluded or marked.

- [x] Add liquidity filters.
  - Filters: min open interest, min volume, max bid-ask spread percent, max quote age.
  - Acceptance: surface construction can be restricted to tradable quotes.

- [x] Add midpoint, mark, and last-price selection.
  - Acceptance: user can choose which market price drives computed IV.

- [x] Add crossed/locked market detection.
  - Acceptance: invalid bid/ask rows are flagged and excluded.

- [x] Add parity sanity checks across calls and puts.
  - Acceptance: obvious put-call parity violations are flagged by expiry/strike.

- [x] Add data quality score per expiry and per surface.
  - Acceptance: UI displays number of valid quotes, rejected quotes, and reason buckets.

## Phase 3: Quant Accuracy Upgrades

Goal: make the analytics credible to someone with options/vol experience.

- [x] Add Black-Scholes-Merton tests against more references.
  - Include dividends, rates, near expiry, deep ITM/OTM, and put/call parity.
  - Acceptance: pricing tests cover edge cases currently only lightly tested.

- [x] Add forward price and discount factor framework.
  - Acceptance: surface construction can work in forward moneyness, not only spot moneyness.

- [x] Re-express surface axes as log-moneyness and expiry.
  - Acceptance: chart can toggle strike, moneyness, log-moneyness, and delta axes.

- [x] Add delta-based skew metrics.
  - Metrics: 10d put IV, 25d put IV, ATM IV, 25d call IV, 10d call IV, risk reversal, butterfly.
  - Acceptance: dashboard calculates these per expiry.

- [x] Add term-structure analytics.
  - Metrics: front/back spread, slope, curvature, contango/backwardation flag.
  - Acceptance: term structure tab shows ATM IV by expiry with slope metrics.

- [x] Add no-arbitrage checks.
  - Checks: call monotonicity in strike, convexity in strike, calendar monotonicity where applicable, bounds by option type.
  - Acceptance: violations are surfaced and optionally excluded.

- [x] Add arbitrage-aware smoothing.
  - Suggested first step: robust smoothing with penalties, then later SVI/SSVI.
  - Acceptance: fitted surface is smoother without hiding raw data points.

- [x] Add SVI smile calibration by expiry.
  - Acceptance: raw smile, fitted SVI curve, parameters, and fit error are visible.

- [x] Add SSVI or arbitrage-constrained global surface calibration.
  - Acceptance: cross-expiry surface has documented constraints and fit diagnostics.

- [x] Add fit diagnostics.
  - Metrics: RMSE, MAE, max error, bid/ask fit rate, rejected points.
  - Acceptance: every fitted surface has fit-quality metadata.

- [x] Add raw-vs-fitted residual charts.
  - Acceptance: user can see where model over/under-fits.

- [x] Add local volatility approximation.
  - Suggested: Dupire local vol with strong warnings and smoothing requirements.
  - Acceptance: local vol tab is disabled unless data quality and smoothing pass.

- [x] Add realized volatility estimators.
  - Estimators: close-to-close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang.
  - Acceptance: UI compares realized vol to implied vol by window.

- [x] Add IV rank and IV percentile.
  - Acceptance: computed from stored/historical snapshots, not random distribution samples.

- [x] Add expected move calculations.
  - Acceptance: expected move by expiry can use ATM straddle or ATM IV approximation.

- [x] Add event awareness.
  - Events: earnings, FOMC, CPI, dividends.
  - Acceptance: term structure can annotate event expiries.

- [x] Add vol-of-vol and surface-change analytics.
  - Acceptance: dashboard can show IV changes versus previous snapshot.

- [x] Add surface shock analysis.
  - Scenarios: parallel vol shift, skew steepening/flattening, term twist, spot move.
  - Acceptance: user can see Greek/P&L impact of vol shocks.

- [x] Add option price decomposition.
  - Components: intrinsic value, time value, carry, implied vol contribution.
  - Acceptance: chain table can explain option price anatomy.

- [x] Add American option model support.
  - Suggested: binomial tree or Barone-Adesi Whaley approximation.
  - Acceptance: user can compare European BSM vs American approximation for dividend names.

- [x] Add model selection.
  - Models: BSM, BSM with dividends, binomial, Heston placeholder/calibration later.
  - Acceptance: model choice is explicit and visible in chart metadata.

- [x] Add Heston calibration research module.
  - Acceptance: calibration works on stored snapshots with clear fit errors and warnings.

- [x] Add SABR for index/rates-style smiles if relevant.
  - Acceptance: optional module, not forced into equity UI.

- [x] Add Greeks by contract from computed IV.
  - Acceptance: chain rows show contract-level Greeks using the same model/rate/dividend assumptions.

- [x] Add second-order Greeks.
  - Greeks: vanna, volga/vomma, charm, speed, color.
  - Acceptance: advanced Greeks tab can be toggled on.

- [x] Add Greek unit consistency tests.
  - Acceptance: vega, theta, rho units are documented and validated.

- [x] Investigate Vega diagnostic warning.
  - Current diagnostic says PLTR ATM 75 percent vol vega "seems too low"; determine if the warning threshold or units are wrong.
  - Acceptance: diagnostic reflects the project's documented vega convention.

## Phase 4: Advanced Features That Make The Project Stand Out

These are the "insane project" upgrades.

- [x] Real-time surface tape.
  - Record each surface snapshot and animate surface evolution through the day.
  - Acceptance: user can replay surface changes by timestamp.

- [x] Surface change heatmap.
  - Show current IV minus previous close, previous hour, or previous refresh.
  - Acceptance: surface deltas are available in vol points and percent change.

- [x] Rich/cheap scanner.
  - Rank options by residual to fitted surface, bid/ask liquidity, and z-score.
  - Acceptance: scanner outputs candidates with explainable reasons.

- [x] Relative value dashboard.
  - Compare two symbols or sectors by ATM IV, skew, term structure, and realized spread.
  - Acceptance: pair comparison view supports normalized overlays.

- [x] Cross-sectional vol map.
  - Show selected universe by IV rank, IV percentile, skew, term slope, realized/implied spread.
  - Acceptance: user can sort and filter opportunities across many symbols.

- [x] Earnings vol event engine.
  - Estimate implied earnings move, compare to historical moves, show post-event crush.
  - Acceptance: earnings symbols have a dedicated event card.

- [x] Strategy builder.
  - Strategies: straddle, strangle, vertical, calendar, diagonal, butterfly, condor, risk reversal.
  - Acceptance: user can create a strategy and see payoff, Greeks, breakevens, max profit/loss.

- [x] Vol surface-aware strategy pricing.
  - Price each leg using fitted IV for that strike/expiry instead of one flat IV.
  - Acceptance: strategy builder uses current surface interpolation.

- [x] Scenario engine for strategies.
  - Axes: spot, time, vol shift, skew shift.
  - Acceptance: P&L heatmaps update from scenario inputs.

- [x] Portfolio upload/import.
  - Support CSV position import with symbol, expiry, strike, type, quantity, cost.
  - Acceptance: risk tab calculates aggregate Greeks and scenario P&L.

- [x] Portfolio optimization.
  - Optimize hedges for delta-neutral, vega-neutral, theta target, or max loss constraint.
  - Acceptance: suggestions include contract, size, estimated cost, and trade-offs.

- [x] Alerts system.
  - Alerts: IV rank threshold, skew steepening, surface fit error, data stale, rich/cheap residual.
  - Acceptance: alerts can be configured and logged locally.

- [x] Watchlist presets.
  - Presets: mega-cap tech, indices, high beta, financials, earnings this week.
  - Acceptance: user can switch universes quickly.

- [x] Saved workspaces.
  - Save selected symbols, filters, model settings, and chart layout.
  - Acceptance: user can reload a workspace from local config.

- [x] Snapshot comparison.
  - Compare two saved snapshots side by side.
  - Acceptance: surface, skew, term, and scanner deltas are shown.

- [x] Backtesting framework.
  - Test strategies triggered by IV rank, skew, term structure, or residual signals.
  - Acceptance: backtest tab reports return, drawdown, hit rate, Sharpe, turnover, transaction costs.

- [x] Transaction cost model.
  - Include bid/ask spread, slippage, commissions, and assignment/exercise assumptions.
  - Acceptance: backtests and strategy P&L use explicit costs.

- [x] Paper-trading simulator.
  - Acceptance: positions can be entered, marked, and tracked without broker integration.

- [x] Broker integration abstraction.
  - Future support for read-only positions first; trading actions should remain disabled until explicitly designed.
  - Acceptance: no accidental live trading functionality.

- [x] Notebook export.
  - Export current analysis to a reproducible Jupyter notebook or HTML report.
  - Acceptance: report includes data timestamp and model assumptions.

- [x] Research report generator.
  - Generate symbol-specific surface summary with charts and diagnostics.
  - Acceptance: one-click local report file from current dashboard state.

- [x] ML anomaly detector.
  - Detect unusual surface moves or residuals using historical features.
  - Acceptance: model is trained/evaluated on local snapshots with explainable features.

- [x] Vol regime classifier.
  - Cluster market regimes based on realized vol, implied vol, skew, term slope, correlations.
  - Acceptance: regime label appears with confidence and historical analogs.

- [x] Forecasting module.
  - Forecast realized vol or IV changes using baseline models first.
  - Acceptance: compares naive baseline, GARCH, and optional ML model.

- [x] News/event overlay.
  - Use only trusted sources and show source links.
  - Acceptance: events explain surface jumps without cluttering the chart.

- [x] WebSocket/async refresh engine.
  - Acceptance: data updates do not block Streamlit rerenders or freeze the UI.

- [x] Multi-page app architecture.
  - Acceptance: pages load independently and share a central state/data service.

## Phase 5: Data Science And Engineering Quality

- [x] Introduce typed configuration.
  - Suggested: Pydantic settings or dataclasses.
  - Acceptance: config has validation, defaults, and environment overrides.

- [x] Add structured logging.
  - Acceptance: logs include source, symbol, provider, latency, cache hit, fallback reason.

- [x] Add performance timing.
  - Acceptance: dashboard can show slowest data/calculation steps.

- [x] Add deterministic random seed only for demo mode.
  - Acceptance: demo mode is stable across reruns unless explicitly randomized.

- [x] Move demo data into a named demo provider.
  - Acceptance: no random demo logic lives in dashboard rendering code.

- [x] Add fixtures for option chains.
  - Acceptance: tests can run without network.

- [x] Add provider contract tests.
  - Acceptance: every provider returns the canonical models and metadata.

- [x] Add surface builder tests.
  - Cases: empty data, missing columns, sparse expiries, bad IVs, extreme moneyness, NaN fills.
  - Acceptance: each fallback path is covered.

- [x] Add chain-cleaning tests.
  - Acceptance: invalid quotes are rejected for clear documented reasons.

- [x] Add dashboard AppTest coverage.
  - Acceptance: app renders default state, no-symbol state, synthetic mode, and provider failure mode.

- [x] Add lint and format workflow.
  - Current repo has Ruff config but requirements include Black/Flake8.
  - Acceptance: one command runs format/lint/tests consistently.

- [x] Update CI.
  - Acceptance: CI runs pytest, compileall, lint, and dashboard smoke test.

- [x] Add dependency pin strategy.
  - Acceptance: project supports reproducible installs with either `requirements.lock` or `uv.lock`.

- [x] Exclude generated logs/cache from git.
  - Acceptance: `.gitignore` covers smoke logs, Streamlit cache, generated snapshots if desired, and pycache.

- [x] Add README screenshots after UI redesign.
  - Acceptance: README shows real app state and explains live vs demo.

- [x] Add architecture diagram.
  - Acceptance: docs show providers -> normalized data -> quant engine -> dashboard.

- [ ] Add glossary.
  - Terms: IV, DTE, moneyness, log-moneyness, delta, skew, risk reversal, butterfly, IV rank, SVI.
  - Acceptance: concise docs for non-expert reviewers without cluttering app UI.

## Phase 6: Specific UI Components To Build

- [ ] `SurfaceWorkspace`
  - Main surface view with 3D surface, 2D heatmap, raw points overlay, fit residuals, and axis toggles.

- [ ] `ChainExplorer`
  - Option-chain grid with filters, IV/G Greeks, liquidity flags, and row-level details.

- [ ] `SkewLab`
  - Smile by expiry, delta skew metrics, risk reversal/butterfly table, raw vs fitted curve.

- [ ] `TermStructurePanel`
  - ATM IV curve, realized vol overlays, event markers, front/back spread.

- [ ] `DataQualityPanel`
  - Source, timestamp, cache age, rejected rows, no-arbitrage violations, fit errors.

- [ ] `ScannerPanel`
  - Universe-level ranking by IV rank, skew, term slope, residual rich/cheap score.

- [ ] `StrategyBuilder`
  - Leg editor, payoff chart, Greeks table, scenario controls, surface-based pricing.

- [ ] `PortfolioRiskPanel`
  - Position import, aggregate Greeks, scenario P&L, concentration and hedge suggestions.

- [ ] `DiagnosticsPanel`
  - Provider health, latency, exceptions, data-source capability, latest logs.

- [ ] `ReportExportPanel`
  - Export current symbol analysis to HTML or notebook.

## Suggested File/Module Roadmap

```text
src/
  dashboard/
    __init__.py
    app_shell.py
    state.py
    theme.py
    charts.py
    components/
      controls.py
      kpi_strip.py
      market_table.py
      surface_workspace.py
      chain_explorer.py
      skew_lab.py
      term_structure.py
      data_quality.py
      scanner.py
      strategy_builder.py
      diagnostics.py
  data/
    models.py
    providers/
      base.py
      yfinance_provider.py
      demo_provider.py
      csv_provider.py
    cache.py
    snapshots.py
  quant/
    rates.py
    dividends.py
    options.py
    greeks.py
    surface.py
    svi.py
    arbitrage.py
    realized_vol.py
    strategies.py
    scenarios.py
  services/
    market_data_service.py
    surface_service.py
    scanner_service.py
tests/
  data/
  quant/
  dashboard/
```

## Immediate High-Impact Sprint

If only one sprint is available, do these in order:

1. Make live/synthetic/fallback status honest across the UI.
2. Create canonical data models for market snapshots and option quotes.
3. Build a real yfinance option-chain provider with cache and source metadata.
4. Replace random portfolio/correlation/P&L sections with real calculations or disabled demo labels.
5. Refactor `app.py` into dashboard modules.
6. Redesign the first screen around command bar, KPI strip, surface workspace, and data-quality panel.
7. Add liquidity filters, no-arbitrage checks, and surface fit diagnostics.
8. Add term structure, skew metrics, IV rank, realized vol comparison, and snapshot persistence.
9. Add AppTest coverage and CI smoke tests.
10. Add scanner and strategy builder as the first standout features.

## Acceptance Definition For "Professional"

The upgrade is not done until:

- A user can tell exactly what is live, delayed, cached, synthetic, or unavailable.
- Every chart has units, timestamps, source metadata, and clean hover labels.
- The app renders without Streamlit deprecation warnings.
- The main dashboard fits a serious quant workflow: select symbol, inspect data quality, inspect surface, inspect skew/term, inspect chain, evaluate risk.
- Placeholder/random metrics are gone from live mode.
- Surface fitting has data quality diagnostics and no-arbitrage checks.
- Tests cover pricing, IV, data cleaning, surface construction, provider normalization, and dashboard render states.
- README accurately describes what the app can and cannot do.
