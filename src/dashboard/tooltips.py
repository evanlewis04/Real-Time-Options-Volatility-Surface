"""Compact help text for workstation controls and table columns."""

from __future__ import annotations


CONTROL_HELP = {
    "universe": "Symbols used for market table, correlation, and comparison panels.",
    "primary_underlying": "Underlying used for surface, chain, skew, and term views.",
    "show_3d_surface": "Toggle the 3D annualized-IV surface.",
    "show_correlations": "Use aligned daily returns from yfinance closes.",
    "show_chain": "Show normalized yfinance option quotes after quality filters.",
    "auto_refresh": "Rerun the dashboard on the selected interval.",
    "refresh_interval": "Seconds between automatic reruns.",
    "max_spread_pct": "Exclude quotes with bid-ask spread above this fraction of mid.",
    "min_open_interest": "Minimum open interest required in the chain table.",
    "min_volume": "Minimum reported contract volume required for surface and chain inputs.",
    "max_quote_age_days": "Maximum quote age for rows with provider timestamps; untimestamped rows are kept.",
    "option_price_source": "Market price used to recompute IV for surface construction.",
}


KPI_HELP = {
    "Spot": "Latest underlying price from the active provider.",
    "ATM IV": "Nearest-strike annualized implied volatility on the fitted surface.",
    "Term Spread": "Back ATM IV minus front ATM IV, in volatility points.",
    "Surface Points": "Number of fitted grid points in the displayed surface.",
    "Contracts": "Valid option rows used or summarized after filtering.",
    "Median Spread": "Median option bid-ask spread divided by mid price.",
}


COLUMN_HELP = {
    "Spot": "Underlying price in dollars.",
    "30D IV": "Approximate 30-day annualized implied volatility.",
    "60D IV": "Approximate 60-day annualized implied volatility.",
    "90D IV": "Approximate 90-day annualized implied volatility.",
    "30D Rate": "Interpolated annualized risk-free rate for a 30 calendar-day maturity.",
    "30D Div Yield": "Effective dividend yield for a 30 calendar-day maturity.",
    "Action Warnings": "Upcoming splits, dividends, or other corporate actions surfaced in data warnings.",
    "Delta": "Approximate option delta, dollars of option per dollar of spot.",
    "Gamma": "Approximate delta change per dollar of spot.",
    "Theta/day": "Approximate option time decay in dollars per calendar day.",
    "Vega/1%": "Approximate option price change for a one volatility-point move.",
    "Contracts": "Count of valid option contracts in the current source.",
    "Volume": "Sum of reported option-chain contract volume.",
    "Mode": "Live, delayed, synthetic, fallback, or unavailable data mode.",
    "IV Source": "Source used for representative implied-volatility fields.",
    "type": "Call or put.",
    "expiration": "Contract expiration date.",
    "daysToExpiration": "Calendar days to expiration.",
    "strike": "Contract strike in dollars.",
    "moneyness": "Strike divided by spot.",
    "bid": "Best reported bid in dollars.",
    "ask": "Best reported ask in dollars.",
    "mid": "Strict bid/ask midpoint when the market has a positive bid and ask above bid.",
    "mark": "Provider mark when available, otherwise midpoint or last price fallback.",
    "last": "Last reported trade price in dollars.",
    "selectedMarketPrice": "Market price selected to recompute implied volatility.",
    "selectedPriceSource": "Row-level source used for the selected market price.",
    "openInterest": "Open contracts reported by the data provider.",
    "impliedVolatility": "Provider implied volatility, annualized.",
    "computedIV": "Implied volatility recomputed from the selected market price, rate, and dividend assumptions.",
    "parityViolation": "Call and put at the same expiry/strike fail the put-call parity sanity tolerance.",
    "parityError": "Observed call-minus-put price difference minus theoretical parity difference.",
    "riskFreeRate": "Interpolated annualized risk-free rate for this contract expiry.",
    "effectiveDividendYield": "Continuous dividend yield used by BSM for this expiry, including discrete dividend adjustment.",
    "discreteDividendAmount": "Undiscounted cash dividends with ex-dates before this contract expires.",
    "quoteQuality": "Provider quote quality label such as bid_ask, stale_bid_ask, or last_only.",
    "isCrossedMarket": "Bid is greater than ask; these rows are flagged and excluded before surface fitting.",
    "isLockedMarket": "Bid equals ask with both sides positive; these rows are flagged and excluded before surface fitting.",
    "quoteAgeSeconds": "Seconds between provider quote timestamp and snapshot normalization time.",
    "bidAskSpreadPct": "Bid-ask spread divided by mid price.",
}


def column_help(name: str) -> str | None:
    """Return compact help text for a displayed column."""
    return COLUMN_HELP.get(name)
