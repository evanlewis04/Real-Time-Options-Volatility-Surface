# Glossary

This glossary defines the main options-volatility terms used by the dashboard.
It is intentionally concise so reviewers can understand the workstation without
adding long explanatory text to the app itself.

## Core Terms

**IV**  
Implied volatility. The annualized volatility input that makes an option-pricing
model match an observed option price. In this project IV can be provider-reported
or recomputed from bid/ask midpoint, mark, or last price; the selected source is
shown in metadata.

**DTE**  
Days to expiration. The calendar-day count from the snapshot date to the option
expiry. DTE is used for term structure, expected move, and time-to-expiry inputs
in pricing models.

**Moneyness**  
Strike divided by spot price. A value below 1.0 means the strike is below spot;
a value above 1.0 means the strike is above spot. The dashboard uses moneyness
to compare option smiles across symbols with different stock prices.

**Log-Moneyness**  
The natural log of strike divided by forward or spot price. Log-moneyness makes
upside and downside distances more symmetric than raw strike or simple
moneyness, which helps when fitting and comparing smiles.

**Delta**  
The approximate change in option value for a one-dollar change in the underlying
price. Calls usually have positive delta and puts usually have negative delta.
The dashboard can use delta as a surface axis and shows contract-level Greeks
when enough quote and model inputs are available.

**Skew**  
The shape of implied volatility across strikes or deltas for the same expiry.
Equity options often show higher IV for downside puts than upside calls. Skew
metrics help summarize that shape without reading every contract row.

**Risk Reversal**  
A skew metric comparing out-of-the-money call IV with out-of-the-money put IV at
the same absolute delta, commonly 25-delta. In this project the 25-delta risk
reversal is call IV minus put IV, so negative values indicate richer downside
put volatility.

**Butterfly**  
A curvature metric comparing wing IV with ATM IV, commonly using 25-delta put
and call wings. In this project the 25-delta butterfly is the average of the
25-delta put and call IVs minus ATM IV.

**IV Rank**  
A normalized measure of where current IV sits between its historical minimum
and maximum over the selected lookback. A value near 1.0 means current IV is
near the top of that historical range; a value near 0.0 means it is near the
bottom.

**SVI**  
Stochastic Volatility Inspired smile parameterization. The project calibrates
SVI by expiry to fit a smooth implied-volatility smile while keeping raw quotes,
fit errors, and rejected points visible for review.

## Provenance Terms

**Live/Delayed**  
Market data sourced from a provider such as yfinance. The app labels this mode
as delayed when the provider path is live enough to use but not guaranteed to be
real-time exchange data.

**Synthetic**  
Deterministic demo data generated locally when real option-chain data is not
available. Synthetic surfaces are useful for offline testing and UI continuity
but should not be interpreted as market quotes.

**Fallback**  
A recovery path used when a provider or calculation cannot produce a usable live
or synthetic result. Fallback mode should always include a visible reason in the
dashboard metadata or Diagnostics tab.

**Data Quality Score**  
A percentage-style score summarizing usable quotes versus rejected quotes. It is
computed from documented rejection buckets such as invalid strike, stale quote,
crossed market, low liquidity, or wide bid/ask spread.
