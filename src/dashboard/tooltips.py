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
    "mid": "Bid/ask midpoint, or last price when no valid market exists.",
    "last": "Last reported trade price in dollars.",
    "openInterest": "Open contracts reported by the data provider.",
    "impliedVolatility": "Provider implied volatility, annualized.",
    "bidAskSpreadPct": "Bid-ask spread divided by mid price.",
}


def column_help(name: str) -> str | None:
    """Return compact help text for a displayed column."""
    return COLUMN_HELP.get(name)
