# Upgrade Plan: Robust Volatility Surface Fitting

This plan replaces the completed dashboard upgrade backlog. It focuses on the next problem observed in live use: noisy option-chain quotes can distort the fitted volatility surface, especially in short-dated expiries with stale, wide, last-only, or no-arbitrage-violating quotes.

The goal is not to make bad market data look clean. The goal is to build an honest robust surface workflow that:

- Keeps raw market quotes visible.
- Scores and explains quote reliability.
- Builds a more stable fitted surface from weighted, filtered inputs.
- Uses historical priors and ML only as labeled denoising aids.
- Preserves provenance so the dashboard never confuses observed quotes with estimated values.

## Current Observed Issue

Recent AAPL dashboard output showed:

- Data quality score around `58/100`.
- Hundreds of no-arbitrage violations.
- Many rejected rows from expired contracts, extreme moneyness, old quotes, stale last-only quotes, wide bid/ask spreads, and invalid IV.
- Raw IV points as high as roughly `200%` to `300%` against a fitted surface around `20%` to `65%`.
- SVI fit that was usable but noisy.
- SSVI and Heston diagnostics with high RMSE.

This is a data-quality and robust-fitting problem. The dashboard should continue to expose the raw quote cloud, but the primary fitted surface should become less sensitive to unreliable rows.

## Implementation Rules

- Preserve data provenance everywhere: source, timestamp, cache age, selected market price source, filters, fit mode, weighting mode, fallback reason, and model assumptions.
- Keep deterministic offline tests. Use fixed fixtures for quote-quality scoring, no-arbitrage cases, historical priors, and ML predictions.
- Do not use ML predictions as market truth. Label them as denoised estimates or prior-assisted estimates.
- Do not hide bad data. Show raw quotes, rejected rows, excluded rows, weights, and fit residuals.
- Prefer transparent robust methods before adding opaque models.
- Keep the default surface conservative. The user should be able to compare raw, standard fit, robust fit, and ML-denoised fit.
- Add feature flags or explicit dashboard controls before changing default behavior.
- Every meaningful quant/data change needs unit tests and at least one dashboard/AppTest coverage point.
- Full verification before closing an implementation session:

```powershell
python -m ruff check src tests diagnostic.py dashboard_connector.py app.py main.py scripts
python -m compileall src tests scripts dashboard_connector.py diagnostic.py app.py main.py
python -m pytest -q
$env:PYTHONPATH='.'; python scripts\healthcheck.py
python diagnostic.py
python scripts\verify.py --skip-healthcheck
python scripts\dashboard_visual_regression.py --port 8536 --output-dir artifacts\dashboard_screenshots --viewports desktop
```

## Phase 0: Baseline Investigation And Reproducible Fixtures

- [x] Capture a deterministic noisy-chain fixture.
  - Add a fixture that resembles the observed AAPL issue: stale rows, wide spreads, last-only rows, no-arbitrage violations, extreme moneyness, and several raw IV spikes.
  - Suggested files: `tests/fixtures/noisy_option_chain.py`, `tests/test_robust_surface_fixtures.py`.
  - Acceptance: fixture produces stable row counts and reason buckets without network access.

- [x] Add a clean-chain control fixture.
  - Add a comparable fixture with smooth IV smiles, reasonable spreads, and no material no-arbitrage violations.
  - Acceptance: clean fixture quality score is high and robust fit should closely match standard fit.

- [x] Add a reproducible before/after diagnostic script.
  - Script should run standard fit against noisy and clean fixtures and print quality, fit RMSE, residual quantiles, and excluded-row counts.
  - Suggested file: `scripts/compare_surface_fit_modes.py`.
  - Acceptance: command runs offline and emits deterministic JSON or table output.

- [x] Document current fit behavior.
  - Add a short note explaining why raw quote clouds can look very different from fitted surfaces.
  - Suggested file: `docs/surface_quality.md`.
  - Acceptance: README or dashboard glossary links to the note without adding bulky in-app text.

## Phase 1: Quote Reliability Scoring

- [x] Create a row-level `QuoteReliabilityScore`.
  - Score each option row from `0.0` to `1.0`.
  - Inputs should include bid/ask spread, quote age, last-only status, volume, open interest, no-arbitrage flags, moneyness distance, provider IV validity, selected market price source, and expiry.
  - Suggested file: `src/quant/quote_quality.py`.
  - Acceptance: deterministic tests prove reliable ATM rows score higher than stale/wide/extreme/no-arb rows.

- [x] Add row-level reason labels.
  - Keep both hard rejection reasons and soft penalty reasons.
  - Example soft penalties: `wide_spread_penalty`, `stale_quote_penalty`, `low_liquidity_penalty`, `extreme_moneyness_penalty`, `last_only_penalty`.
  - Acceptance: dashboard metadata can explain why a quote received a low weight.

- [x] Expose reliability fields in normalized chain snapshots.
  - Add columns such as `quoteReliabilityScore`, `fitWeight`, `fitPenaltyReasons`, and `fitEligible`.
  - Suggested files: `dashboard_connector.py`, `src/data/models.py`.
  - Acceptance: `get_options_chain_snapshot` includes the new fields and preserves existing columns.

- [x] Add expiry-level quality summaries.
  - Summarize median score, low-score count, fit-eligible count, excluded count, and dominant penalty reasons by expiry.
  - Acceptance: metadata includes stable expiry summaries for the DataQualityPanel.

## Phase 2: Stricter Fit Eligibility And Controls

- [x] Separate display eligibility from fit eligibility.
  - The option chain grid can display imperfect rows, but the surface fit should use stricter eligibility.
  - Acceptance: rejected-for-fit rows remain visible in the chain with reason labels.

- [x] Add configurable fit filters.
  - Controls: max spread, max quote age, min volume, min open interest, moneyness band, max raw IV, no-arb exclusion policy, last-only policy.
  - Suggested files: `src/config/settings.py`, `src/dashboard/app_shell.py`.
  - Acceptance: fit filters are distinct from chain display filters and included in provenance.

- [x] Exclude no-arbitrage violators from standard fit by default.
  - Current behavior already excludes some surface rows; make the policy explicit and test-covered.
  - Acceptance: metadata reports `no_arbitrage_excluded_count`, `fit_eligible_count`, and `fit_excluded_count`.

- [x] Add a strict preset for poor data days.
  - Presets: `Standard`, `Strict`, `Diagnostic Raw`.
  - `Strict` should reduce stale, wide, last-only, and no-arb rows aggressively.
  - `Diagnostic Raw` should show what happens with minimal filtering but label it clearly.
  - Acceptance: AppTest can select the preset without exceptions.

## Phase 3: Weighted Robust SVI And SSVI Fitting

- [x] Add weighted SVI calibration.
  - Modify per-expiry SVI fitting to accept row weights.
  - Use quote reliability and liquidity as default weights.
  - Suggested file: `src/quant/svi.py`.
  - Acceptance: high-weight clean rows influence fit more than low-weight outliers in deterministic tests.

- [x] Add robust loss support.
  - Support at least `linear`, `huber`, and `soft_l1` loss modes through SciPy least squares.
  - Acceptance: noisy fixture robust RMSE and residual quantiles improve without materially changing clean fixture output.

- [x] Add weighted global SSVI calibration.
  - Extend SSVI fit to use row weights and robust loss.
  - Acceptance: global fit diagnostics include weight mode, loss mode, weighted RMSE, unweighted RMSE, and constraints.

- [x] Add residual clipping diagnostics.
  - Do not silently remove outliers. Report clipped/downweighted rows, residual thresholds, and impact on fit.
  - Acceptance: dashboard can show clipped/downweighted counts and top residual rows.

- [x] Preserve standard fit as a comparison mode.
  - Fit modes: `Standard SVI`, `Robust SVI`, `Robust SSVI`, and later `ML Denoised`.
  - Acceptance: user can compare standard and robust diagnostics from the same quote set.

## Phase 4: Historical Prior And Surface Stabilization

- [x] Build a historical surface prior loader.
  - Use recent persisted snapshots to construct a prior grid by moneyness/log-moneyness and DTE.
  - Suggested files: `src/quant/surface_prior.py`, `src/data/snapshots.py`.
  - Acceptance: loader returns deterministic priors from fixture snapshots and refuses stale/insufficient history.

- [x] Add prior blending for poor-quality current data.
  - Blend current robust fit with recent prior when quote quality is low.
  - Blend weight should depend on current data quality, snapshot recency, and overlap.
  - Acceptance: metadata reports prior source, prior age, blend weight, overlap count, and whether prior was applied.

- [x] Add jump detection before blending.
  - If current clean quotes strongly indicate a true surface move, do not over-anchor to history.
  - Acceptance: tests cover both noisy false spikes and genuine broad IV shifts.

- [x] Add prior comparison charts.
  - Show current robust fit, prior surface, and current-minus-prior heatmap.
  - Acceptance: dashboard labels prior-assisted values as estimates.

## Phase 5: ML Denoising Prototype

- [x] Define a surface ML feature set.
  - Features: log-moneyness, moneyness, DTE, expiry bucket, option type, bid/ask spread, quote age, volume, open interest, selected price source, forward moneyness, rates, dividends, event flags, historical IV prior, and raw IV.
  - Suggested file: `src/ml/surface_features.py`.
  - Acceptance: feature builder is deterministic and handles missing fields.

- [x] Implement a baseline nonparametric denoiser.
  - Start with `ExtraTreesRegressor` or `HistGradientBoostingRegressor` if available, trained only on local historical snapshots and/or deterministic fixtures.
  - Suggested file: `src/ml/surface_denoiser.py`.
  - Acceptance: model can fit fixture data offline and produce bounded IV predictions.

- [x] Add Gaussian Process or kernel smoother research mode.
  - Use log-moneyness and DTE coordinates with uncertainty estimates if dependencies are available.
  - Keep this optional and clearly labeled as research.
  - Acceptance: if dependency is unavailable, module returns a graceful unavailable payload.

- [x] Add uncertainty output.
  - Denoised surfaces should expose prediction uncertainty or confidence bands.
  - Acceptance: dashboard can distinguish high-confidence smooth regions from extrapolated regions.

- [x] Add model persistence for local experiments.
  - Store model metadata, feature schema, training snapshot range, validation metrics, and provenance.
  - Acceptance: saved model can be loaded deterministically and refuses incompatible schemas.

- [x] Keep ML off by default at first.
  - Add explicit `ML Denoised` mode but default to robust deterministic fit until validation is strong.
  - Acceptance: default dashboard behavior remains explainable and deterministic.

## Phase 6: Arbitrage-Aware Surface Repair

- [x] Add post-fit static arbitrage checks.
  - Check calendar monotonicity, butterfly convexity in total variance, positive vols, and smoothness bounds.
  - Suggested file: `src/quant/surface_arbitrage.py`.
  - Acceptance: test fixtures identify calendar and convexity violations.

- [x] Add repair suggestions before automated repair.
  - Report where the surface violates constraints and which input rows likely caused it.
  - Acceptance: DataQualityPanel shows violation locations and likely causes.

- [x] Implement conservative repair mode.
  - Apply minimal smoothing or projection only when enabled, and label repaired regions.
  - Acceptance: repaired surface reduces violations while preserving provenance and raw data visibility.

- [x] Compare raw, robust, prior-assisted, and repaired surfaces.
  - Add diagnostics for RMSE, weighted RMSE, arbitrage violations, smoothness, and residual tail risk.
  - Acceptance: fit comparison table is deterministic in tests.

## Phase 7: Dashboard Workflow Updates

- [x] Add `Fit Mode` controls to SurfaceWorkspace.
  - Modes: `Standard`, `Robust`, `Prior Assisted`, `ML Denoised`, `Diagnostic Raw`.
  - Acceptance: selected mode updates charts, captions, and provenance.

- [x] Add a quote reliability overlay.
  - Raw quote points should be colored or sized by reliability score.
  - Acceptance: low-reliability outliers are visually distinct from clean quotes.

- [x] Add a fit comparison panel.
  - Table: fit mode, eligible rows, excluded rows, weighted RMSE, unweighted RMSE, no-arb violations, prior weight, ML uncertainty, timestamp.
  - Acceptance: all values are sourced from metadata, not recomputed ad hoc in the UI.

- [x] Improve DataQualityPanel for actionability.
  - Show top penalty reasons, worst expiries, worst residual contracts, no-arb violation counts, and suggested stricter preset.
  - Acceptance: the user can understand why the score changed from yesterday.

- [x] Add an alert when quality materially drops.
  - Compare current quality score and reason buckets with prior snapshots.
  - Acceptance: dashboard can say whether a surface shape change is likely data-quality driven.

- [x] Add exports for fit diagnostics.
  - CSV/JSON export for row weights, residuals, fit diagnostics, and provenance.
  - Acceptance: exported payload can reproduce the selected fit mode offline.

## Phase 8: Validation And Backtesting

- [x] Add fit-mode validation metrics.
  - Metrics: out-of-sample residuals by expiry, residual quantiles, stability versus prior day, no-arb violation rate, and smoothness penalties.
  - Acceptance: metrics run offline on fixture snapshots.

- [x] Backtest robust fit against stored snapshots.
  - Compare standard fit versus robust/prior-assisted fits over historical local snapshots.
  - Acceptance: report shows when robust fit improves stability and when it hides real moves.

- [x] Validate rich/cheap scanner under noisy data.
  - Scanner should use selected fit mode and quote reliability.
  - Acceptance: noisy outliers do not dominate scanner candidates unless explicitly shown as low-confidence.

- [x] Add regression tests for yesterday-versus-today comparisons.
  - Use two deterministic snapshots: one clean, one noisy.
  - Acceptance: dashboard flags shape change as likely data-quality driven when reason buckets deteriorate.

## Phase 9: Documentation And Operating Guidance

- [x] Write a surface quality interpretation guide.
  - Explain quality score, fit eligibility, weights, no-arb violations, robust fit, prior-assisted fit, and ML-denoised fit.
  - Suggested file: `docs/surface_quality.md`.
  - Acceptance: concise enough for a user to act on during live analysis.

- [x] Add recommended filter presets.
  - Document when to use Standard, Strict, Diagnostic Raw, Prior Assisted, and ML Denoised.
  - Acceptance: README links to the guide.

- [x] Add provenance examples.
  - Include examples of good data, noisy data, prior-assisted data, and ML-denoised data.
  - Acceptance: examples use deterministic fixture output.

- [x] Update handoff workflow.
  - Future sessions should read this file, implement the next unchecked item, run targeted tests plus full verification for completed phases, update `SESSION_HANDOFF.md`, and commit.

## Suggested Implementation Order

1. Phase 0 fixture work.
2. Phase 1 quote reliability scoring.
3. Phase 2 fit eligibility and strict preset.
4. Phase 3 weighted robust SVI.
5. Phase 7 dashboard controls for Standard versus Robust.
6. Phase 4 historical prior blending.
7. Phase 5 ML denoising prototype.
8. Phase 6 arbitrage-aware repair.
9. Phase 8 validation/backtesting.
10. Phase 9 documentation.

## Definition Of Done

- The dashboard can explain why today's surface differs from yesterday's.
- The user can switch between raw/standard/robust/prior-assisted/ML-denoised views.
- Low-quality quotes are visible but do not silently dominate the primary fitted surface.
- The selected surface mode has complete provenance and deterministic diagnostics.
- Full verification passes with no new unexplained warnings.
