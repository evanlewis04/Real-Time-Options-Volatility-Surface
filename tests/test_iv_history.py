from datetime import datetime, timedelta

import pytest

from src.data.models import MarketDataSnapshot, OptionQuote
from src.data.snapshots import save_snapshot
from src.quant.iv_history import atm_iv_from_chain, iv_rank_percentile_from_snapshots


def _snapshot(timestamp: datetime, iv: float) -> MarketDataSnapshot:
    return MarketDataSnapshot(
        symbol="AAPL",
        spot=100.0,
        spot_timestamp=timestamp,
        chain_timestamp=timestamp,
        expirations=(datetime(2026, 6, 19),),
        options=(
            OptionQuote(
                contract=f"AAPL{timestamp.strftime('%H%M%S')}C00100000",
                type="call",
                strike=100.0,
                expiry=datetime(2026, 6, 19),
                dte=30,
                computed_iv=iv,
                raw_iv=iv,
            ),
        ),
        source="fixture",
        mode="Stored",
        raw_rows=1,
        valid_rows=1,
        rejected_rows=0,
    )


def test_iv_rank_percentile_uses_persisted_snapshots(tmp_path):
    base = datetime(2026, 5, 1, 10, 0, 0)
    for offset, iv in enumerate((0.20, 0.25, 0.30)):
        save_snapshot(_snapshot(base + timedelta(seconds=offset), iv), tmp_path)

    result = iv_rank_percentile_from_snapshots("AAPL", 0.28, tmp_path)

    assert result["available"]
    assert result["source"] == "persisted_snapshots"
    assert result["observations"] == 3
    assert result["iv_rank"] == pytest.approx(0.8)
    assert result["iv_percentile"] == pytest.approx(2 / 3)


def test_atm_iv_from_chain_selects_near_30d_atm_quote(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 1, 10, 0, 0), 0.24), tmp_path)
    metadata = next(tmp_path.glob("*.metadata.json"))
    from src.data.snapshots import load_snapshot

    chain = load_snapshot(metadata).options_frame()

    assert atm_iv_from_chain(chain, spot=100.0) == pytest.approx(0.24)
