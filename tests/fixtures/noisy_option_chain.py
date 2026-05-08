"""Deterministic noisy and clean option-chain fixtures for robust-fit work."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.data.options_provider import YFinanceOptionsProvider
from src.pricing.black_scholes import BlackScholesModel
from src.quant.arbitrage import apply_no_arbitrage_checks


FIXTURE_NOW = datetime(2026, 5, 8, 10, 0, 0)
FIXTURE_SYMBOL = "AAPL"
FIXTURE_SPOT = 200.0
FIXTURE_RATE = 0.03
FIXTURE_DIVIDEND_YIELD = 0.006
FIXTURE_DTES = (14, 30, 60, 90)
FIXTURE_STRIKES = (150.0, 170.0, 185.0, 200.0, 215.0, 230.0, 250.0)


def clean_option_chain_raw(symbol: str = FIXTURE_SYMBOL, spot: float = FIXTURE_SPOT) -> pd.DataFrame:
    """Return a smooth yfinance-shaped chain with stable provenance fields."""
    rows: list[dict[str, Any]] = []
    for dte in FIXTURE_DTES:
        expiry = FIXTURE_NOW + timedelta(days=dte)
        for strike in FIXTURE_STRIKES:
            iv = _fixture_iv(strike, dte, spot)
            for option_type in ("call", "put"):
                rows.append(_quote_row(symbol, spot, expiry, dte, strike, option_type, iv))
    return pd.DataFrame(rows)


def noisy_option_chain_raw(symbol: str = FIXTURE_SYMBOL, spot: float = FIXTURE_SPOT) -> pd.DataFrame:
    """Return a noisy chain resembling the observed AAPL quote-quality issue."""
    df = clean_option_chain_raw(symbol, spot)

    _set_quote(df, dte=14, strike=185.0, option_type="call", implied_volatility=2.85)
    _set_quote(df, dte=30, strike=200.0, option_type="put", implied_volatility=2.35)
    _set_quote(df, dte=60, strike=215.0, option_type="call", implied_volatility=1.95)

    # Valid, but economically suspicious, rows that should remain visible and
    # be flagged by no-arbitrage diagnostics rather than hidden by the fixture.
    _set_quote(df, dte=30, strike=150.0, option_type="call", price=205.0, implied_volatility=0.72)
    _set_quote(df, dte=30, strike=200.0, option_type="call", price=7.0, implied_volatility=0.31)
    _set_quote(df, dte=30, strike=215.0, option_type="call", price=18.0, implied_volatility=0.64)
    _set_quote(df, dte=60, strike=185.0, option_type="put", price=19.0, implied_volatility=0.58)

    issue_rows = [
        _quote_row(
            symbol,
            spot,
            FIXTURE_NOW + timedelta(days=14),
            14,
            205.0,
            "call",
            0.44,
            quote_time=FIXTURE_NOW - timedelta(days=8),
        ),
        _quote_row(
            symbol,
            spot,
            FIXTURE_NOW + timedelta(days=30),
            30,
            210.0,
            "put",
            0.39,
            quote_time=FIXTURE_NOW - timedelta(days=9),
            bid=0.0,
            ask=0.0,
            price=4.75,
        ),
        _quote_row(
            symbol,
            spot,
            FIXTURE_NOW + timedelta(days=60),
            60,
            207.5,
            "call",
            0.33,
            bid=0.25,
            ask=2.95,
            price=1.60,
        ),
        _quote_row(symbol, spot, FIXTURE_NOW + timedelta(days=90), 90, 48.0, "put", 0.80),
        _quote_row(symbol, spot, FIXTURE_NOW + timedelta(days=90), 90, 540.0, "call", 0.85),
        _quote_row(symbol, spot, FIXTURE_NOW - timedelta(days=1), -1, 200.0, "call", 0.28),
        _quote_row(
            symbol,
            spot,
            FIXTURE_NOW + timedelta(days=30),
            30,
            195.0,
            "put",
            0.36,
            bid=0.0,
            ask=0.0,
            price=3.95,
        ),
        _quote_row(
            symbol,
            spot,
            FIXTURE_NOW + timedelta(days=60),
            60,
            222.5,
            "call",
            0.41,
            bid=0.0,
            ask=0.0,
            price=2.35,
        ),
    ]
    return pd.concat([df, pd.DataFrame(issue_rows)], ignore_index=True)


def normalized_clean_chain() -> pd.DataFrame:
    """Return the clean fixture after the provider normalization boundary."""
    return normalize_fixture_chain(clean_option_chain_raw())


def normalized_noisy_chain() -> pd.DataFrame:
    """Return the noisy fixture after the provider normalization boundary."""
    return normalize_fixture_chain(noisy_option_chain_raw())


def checked_clean_chain() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return clean normalized rows annotated with static no-arbitrage checks."""
    return apply_fixture_no_arbitrage_checks(normalized_clean_chain())


def checked_noisy_chain() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return noisy normalized rows annotated with static no-arbitrage checks."""
    return apply_fixture_no_arbitrage_checks(normalized_noisy_chain())


def normalize_fixture_chain(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize fixture rows with the same offline rules used by the provider."""
    return YFinanceOptionsProvider._normalize(
        raw,
        FIXTURE_SYMBOL,
        FIXTURE_SPOT,
        FIXTURE_NOW,
        max_quote_age_days=5,
        min_open_interest=0,
        min_volume=0,
        max_bid_ask_spread_pct=1.5,
    )


def apply_fixture_no_arbitrage_checks(chain: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run no-arbitrage diagnostics using fixture marks as observed prices."""
    checked = chain.copy()
    checked["selectedMarketPrice"] = pd.to_numeric(checked["mark"], errors="coerce")
    checked["selectedPriceSource"] = checked["markSource"]
    checked["computedIV"] = pd.to_numeric(checked["impliedVolatility"], errors="coerce")
    checked["riskFreeRate"] = FIXTURE_RATE
    checked["dividendYield"] = FIXTURE_DIVIDEND_YIELD
    checked["effectiveDividendYield"] = FIXTURE_DIVIDEND_YIELD
    checked["logMoneyness"] = np.log(pd.to_numeric(checked["strike"], errors="coerce") / FIXTURE_SPOT)
    return apply_no_arbitrage_checks(checked, FIXTURE_SPOT, price_column="selectedMarketPrice")


def fixture_reason_buckets(chain: pd.DataFrame, no_arbitrage_meta: dict[str, Any] | None = None) -> dict[str, int]:
    """Return deterministic combined quality buckets for fixture assertions."""
    buckets = Counter({str(k): int(v) for k, v in chain.attrs.get("rejection_reasons", {}).items()})
    if no_arbitrage_meta:
        rows = int(no_arbitrage_meta.get("no_arbitrage_violation_rows") or 0)
        if rows:
            buckets["no_arbitrage_violation"] += rows
    return dict(sorted((key, value) for key, value in buckets.items() if value))


def _quote_row(
    symbol: str,
    spot: float,
    expiry: datetime,
    dte: int,
    strike: float,
    option_type: str,
    implied_volatility: float,
    *,
    quote_time: datetime | None = None,
    bid: float | None = None,
    ask: float | None = None,
    price: float | None = None,
) -> dict[str, Any]:
    time = max(dte, 1) / 365.0
    model_price = BlackScholesModel.option_price(
        spot,
        strike,
        time,
        FIXTURE_RATE,
        implied_volatility,
        option_type,
        q=FIXTURE_DIVIDEND_YIELD,
    )
    mark = round(float(price if price is not None else model_price), 2)
    if bid is None:
        bid = max(0.01, mark - _spread_width(mark, strike, spot))
    if ask is None:
        ask = max(bid + 0.01, mark + _spread_width(mark, strike, spot))
    suffix = "C" if option_type == "call" else "P"
    return {
        "contractSymbol": f"{symbol}{expiry.strftime('%y%m%d')}{suffix}{int(round(strike * 1000)):08d}",
        "strike": float(strike),
        "lastPrice": mark,
        "bid": round(float(bid), 2),
        "ask": round(float(ask), 2),
        "mark": mark,
        "volume": int(80 + max(dte, 0) + round(abs(strike - spot))),
        "openInterest": int(450 + max(dte, 0) * 3 + round(abs(strike - spot) * 2)),
        "impliedVolatility": round(float(implied_volatility), 4),
        "lastTradeDate": quote_time or (FIXTURE_NOW - timedelta(minutes=12 + int(abs(strike - spot)))),
        "type": option_type,
        "expiration": expiry.strftime("%Y-%m-%d"),
    }


def _set_quote(
    frame: pd.DataFrame,
    *,
    dte: int,
    strike: float,
    option_type: str,
    price: float | None = None,
    implied_volatility: float | None = None,
) -> None:
    expiry = (FIXTURE_NOW + timedelta(days=dte)).strftime("%Y-%m-%d")
    mask = (
        (frame["expiration"] == expiry)
        & (frame["strike"].astype(float) == float(strike))
        & (frame["type"] == option_type)
    )
    if price is not None:
        frame.loc[mask, ["lastPrice", "mark"]] = round(float(price), 2)
        frame.loc[mask, "bid"] = round(max(0.01, price - 0.08), 2)
        frame.loc[mask, "ask"] = round(price + 0.08, 2)
    if implied_volatility is not None:
        frame.loc[mask, "impliedVolatility"] = round(float(implied_volatility), 4)


def _fixture_iv(strike: float, dte: int, spot: float) -> float:
    log_money = np.log(strike / spot)
    term = 0.018 * np.sqrt(dte / 365.0)
    skew = 0.055 * max(-log_money, 0.0)
    smile = 0.10 * log_money**2
    return round(0.205 + term + skew + smile, 4)


def _spread_width(mark: float, strike: float, spot: float) -> float:
    del strike, spot
    return max(mark * 0.018, 0.005)
