"""Deterministic option-chain fixtures for offline provider and surface tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


FIXTURE_NOW = datetime(2026, 5, 8, 10, 0, 0)


def raw_yfinance_option_chain(symbol: str = "AAPL", spot: float = 200.0) -> pd.DataFrame:
    """Return a yfinance-shaped chain with calls and puts across expiries."""
    rows = []
    for dte in (30, 60, 90):
        expiry = (FIXTURE_NOW + timedelta(days=dte)).strftime("%Y-%m-%d")
        for strike in (spot * 0.9, spot, spot * 1.1):
            money = strike / spot
            iv = 0.22 + 0.02 * (dte / 90) + 0.05 * abs(money - 1.0)
            for opt_type, suffix, price_shift in (("call", "C", max(spot - strike, 0)), ("put", "P", max(strike - spot, 0))):
                mid = 4.0 + price_shift + 0.02 * dte + abs(strike - spot) * 0.03
                rows.append(
                    {
                        "contractSymbol": f"{symbol}{(FIXTURE_NOW + timedelta(days=dte)).strftime('%y%m%d')}{suffix}{int(strike * 1000):08d}",
                        "strike": float(strike),
                        "lastPrice": round(mid, 2),
                        "bid": round(mid - 0.05, 2),
                        "ask": round(mid + 0.05, 2),
                        "volume": 100 + dte,
                        "openInterest": 500 + dte,
                        "impliedVolatility": round(iv, 4),
                        "lastTradeDate": FIXTURE_NOW - timedelta(minutes=15),
                        "type": opt_type,
                        "expiration": expiry,
                    }
                )
    return pd.DataFrame(rows)
