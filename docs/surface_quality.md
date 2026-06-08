# Surface Quality And Fitted IV

Raw option-chain quotes are market-data provider observations. Fitted surfaces,
robust fits, prior-assisted surfaces, conservative repairs, ML-denoised values,
and validation metrics are estimates or diagnostics. They help explain the
quote cloud, but they must not be relabeled as traded market observations.

Use this guide during live analysis when the surface looks smooth but the quote
cloud looks messy, or when the scanner highlights rich/cheap contracts from a
low-quality chain.

## Fast Read

Start with the data-quality row and the raw IV points. A high quality score with
few exclusions means the fitted surface is probably describing the visible quote
cloud. A low or falling score means the fit should be treated as a stabilized
estimate until the raw rows explain themselves.

Read these fields together:

- `surface_quality_score`: Chain-level score after normalization, fit
  eligibility, quote reliability, and no-arb exclusions.
- `fit_eligible_count` and `fit_excluded_count`: Rows allowed into or kept out
  of the fit. Excluded rows can still be shown as diagnostics.
- `quality_reason_buckets`: Why rows were rejected or penalized, such as stale
  quotes, wide spreads, extreme moneyness, or no-arb violations.
- `quoteReliabilityScore` and `fitWeight`: Row-level reliability inputs for
  robust fitting and scanner confidence.
- `fit_mode_validation`: Offline-style holdout diagnostics for fit modes:
  out-of-sample residuals, residual quantiles, no-arb violation rate,
  smoothness penalty, and stability versus a prior snapshot.

## Fit Modes

Use `Standard` when quote quality is high and you want the plain unweighted
linear SVI comparison. It is useful as a baseline, not as proof that noisy rows
are trustworthy.

Use `Robust` as the default fitted estimate. It keeps the raw quote cloud
visible while reducing the influence of rows with weak reliability or liquidity.
Robust fitted values are still estimates and are labeled
`current_robust_fit_estimate_not_market_observation`.

Use `Strict` fit filters when the quality score drops, no-arb buckets grow, or
the scanner is dominated by stale, last-only, wide-spread, or extreme-moneyness
rows. Strict mode removes more rows from the fitting input; it does not delete
the diagnostic evidence.

Use `Diagnostic Raw` when investigating provider behavior. This view emphasizes
observed quote points and reliability labels. It is the right mode for asking
"which rows caused this?" rather than "what is the cleanest estimate?"

Use `Prior Assisted` only when current quote quality is poor and recent local
snapshots are available. The blend is labeled
`prior_assisted_fit_estimate_not_market_observation`; it can stabilize a bad
refresh, but it can also damp a real move.

Use `ML Denoised` as research-only. It is off by default and labeled
`ml_denoised_research_estimate_not_market_observation`. Treat it as a
comparison signal, not as market truth.

## Validation Metrics

Run the deterministic validation report with:

```powershell
python scripts\validate_surface_fit_modes.py --json
```

The noisy fixture currently reports:

| Mode | OOS RMSE | No-Arb Rate | Max Adjacent IV Change |
| --- | ---: | ---: | ---: |
| Standard SVI | 0.7130 | 0.7917 | 0.1310 |
| Robust SVI | 0.6400 | 0.1667 | 0.0180 |
| Robust SSVI | 0.6381 | 0.2500 | 0.0078 |

These are validation diagnostics from deterministic fixtures. They do not mean
the robust fitted IVs traded in the market. They mean the robust modes were less
sensitive to the known noisy rows in this fixture.

The same report includes two backtest narratives:

- Clean to noisy: robust fit improves stability while quality buckets
  deteriorate.
- Clean to stable-quality shifted: the report flags a possible hidden real move
  when robust or prior-assisted estimates damp a material change without a
  matching quality deterioration.

## Scanner Guidance

The rich/cheap scanner ranks residuals using the selected fit mode, liquidity,
and `quoteReliabilityScore`. A low-confidence candidate can still appear, but
it should not dominate only because its residual is large. Read the
`confidence_label`, spread, open interest, volume, and reason text before acting
on the candidate.

When the scanner output feels surprising, switch to `Diagnostic Raw`, inspect
the row-level reliability overlay, then rerun with `Strict` filters. If the same
candidate remains high confidence, it is a cleaner relative-value lead. If it
falls to low confidence, treat it as a data-quality diagnostic.

## Provenance Examples

Good data:
Clean fixture rows have quality score `100.0`, no rejection buckets, and low
fit error. Standard and robust fits should be close. The fitted surface remains
a model estimate, but the raw quote cloud supports it.

Noisy data:
The noisy fixture has quality score `90.6`, with buckets including
`no_arbitrage_violation`, `extreme_moneyness`, `old_quote`,
`stale_last_only`, and `wide_bid_ask_spread`. Robust validation improves the
no-arb rate versus standard SVI. Those robust values are stabilized estimates.

Prior-assisted data:
When a prior blend is applied, the output is labeled
`prior_assisted_fit_estimate_not_market_observation`. Use it to compare current
poor-quality quotes with recent local history, not to replace the quote cloud.

ML-denoised data:
ML-denoised output is labeled
`ml_denoised_research_estimate_not_market_observation`, off by default, and
intended for research comparison only.

Validation-derived data:
`fit_mode_validation_diagnostic_not_market_observation` and
`fit_mode_backtest_diagnostic_not_market_observation` summarize fit behavior.
They can justify using Robust or Strict modes, but they are not market prices,
quotes, or traded IVs.
