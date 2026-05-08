from datetime import datetime, timedelta

import pandas as pd

from src.data.options_provider import YFinanceOptionsProvider


def _row(now, expiration, contract, **overrides):
    base = {
        "contractSymbol": contract,
        "strike": 200.0,
        "lastPrice": 8.2,
        "bid": 8.0,
        "ask": 8.4,
        "volume": 120,
        "openInterest": 500,
        "impliedVolatility": 0.24,
        "lastTradeDate": now - timedelta(hours=1),
        "type": "call",
        "expiration": expiration,
    }
    base.update(overrides)
    return base


def test_chain_cleaning_documents_base_rejection_reasons():
    now = datetime(2026, 5, 2, 12, 0, 0)
    expiration = (now + timedelta(days=30)).strftime("%Y-%m-%d")
    expired = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    raw = pd.DataFrame(
        [
            _row(now, expiration, "AAPL260601C00200000"),
            _row(now, expiration, "AAPL260601C00000000", strike=0.0),
            _row(now, expired, "AAPL260501C00200000"),
            _row(now, expiration, "AAPL260601C00205000", lastPrice=0.0, bid=0.0, ask=0.0),
            _row(now, expiration, "AAPL260601C00210000", impliedVolatility=7.5),
            _row(now, expiration, "AAPL260601C00900000", strike=900.0),
        ]
    )

    clean = YFinanceOptionsProvider._normalize(raw, "AAPL", 200.0, now)

    assert clean["contractSymbol"].tolist() == ["AAPL260601C00200000"]
    assert clean.attrs["rejection_reasons"] == {
        "invalid_strike": 1,
        "expired_contract": 1,
        "invalid_mark": 1,
        "invalid_implied_volatility": 1,
        "extreme_moneyness": 1,
    }
    quality = clean.attrs["expiry_quality"]
    assert quality[expiration]["valid_quotes"] == 1
    assert quality[expiration]["rejected_quotes"] == 4
    assert quality[expiration]["reason_buckets"] == {
        "invalid_strike": 1,
        "invalid_mark": 1,
        "invalid_implied_volatility": 1,
        "extreme_moneyness": 1,
    }
    assert quality[expired]["reason_buckets"] == {"expired_contract": 1}


def test_chain_cleaning_keeps_last_only_quotes_when_fresh_and_marked():
    now = datetime(2026, 5, 2, 12, 0, 0)
    expiration = (now + timedelta(days=30)).strftime("%Y-%m-%d")
    raw = pd.DataFrame(
        [
            _row(
                now,
                expiration,
                "AAPL260601P00200000",
                type="put",
                bid=None,
                ask=None,
                lastPrice=7.5,
            )
        ]
    )

    clean = YFinanceOptionsProvider._normalize(raw, "AAPL", 200.0, now)

    assert clean["contractSymbol"].tolist() == ["AAPL260601P00200000"]
    row = clean.iloc[0]
    assert row["quoteQuality"] == "last_only"
    assert row["markSource"] == "last"
    assert row["mark"] == 7.5
    assert clean.attrs["rejection_reasons"] == {}
