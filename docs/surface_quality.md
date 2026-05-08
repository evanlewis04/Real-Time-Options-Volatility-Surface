# Surface Quality And Fitted IV

Raw option-chain quotes are observations from the market data provider. The fitted
surface is an estimate built from selected rows. Those two views can differ
sharply when the chain contains stale quotes, wide spreads, last-only prices,
extreme moneyness, invalid IV, or no-arbitrage violations.

The dashboard should keep the raw quote cloud visible while making the fit
provenance explicit: source, timestamp, selected market price, IV input,
rejection reasons, no-arbitrage exclusions, and fit diagnostics. A smoother
surface is not proof that the market traded smoothly. It is only the model's
best estimate from the rows allowed into the fit.

For the current standard fit, rows with static no-arbitrage violations are
excluded from the fitting input but remain visible in chain diagnostics. Future
robust, prior-assisted, and ML-denoised modes must stay labeled as estimates.
ML output is a denoising aid or prior signal, never market truth.

Use the deterministic Phase 0 comparison script to reproduce the baseline:

```powershell
python scripts\compare_surface_fit_modes.py --json
```

The clean fixture should show high quality and low fit error. The noisy fixture
should show stable rejection buckets, no-arbitrage exclusions, residual tails,
and lower quality before any robust-fitting upgrades are introduced.
