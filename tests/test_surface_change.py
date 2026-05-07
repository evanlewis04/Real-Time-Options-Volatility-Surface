from datetime import datetime

import pandas as pd
import pytest

from dashboard_connector import DashboardConnector
from src.data.models import MarketDataSnapshot, OptionQuote
from src.data.snapshots import save_snapshot
from src.quant.surface_change import atm_iv_vol_of_vol_from_snapshots, surface_change_analytics


EXPIRY = datetime(2026, 6, 19)


def _snapshot(timestamp: datetime, ivs: tuple[float, float, float]) -> MarketDataSnapshot:
    return MarketDataSnapshot(
        symbol="AAPL",
        spot=100.0,
        spot_timestamp=timestamp,
        chain_timestamp=timestamp,
        expirations=(EXPIRY,),
        options=(
            OptionQuote(
                contract=f"AAPL{timestamp:%Y%m%d}C00100000",
                type="call",
                strike=100.0,
                expiry=EXPIRY,
                dte=30,
                computed_iv=ivs[0],
                raw_iv=ivs[0],
            ),
            OptionQuote(
                contract=f"AAPL{timestamp:%Y%m%d}P00100000",
                type="put",
                strike=100.0,
                expiry=EXPIRY,
                dte=30,
                computed_iv=ivs[1],
                raw_iv=ivs[1],
            ),
            OptionQuote(
                contract=f"AAPL{timestamp:%Y%m%d}C00110000",
                type="call",
                strike=110.0,
                expiry=EXPIRY,
                dte=30,
                computed_iv=ivs[2],
                raw_iv=ivs[2],
            ),
        ),
        source="fixture",
        mode="Stored",
        raw_rows=3,
        valid_rows=3,
        rejected_rows=0,
    )


def _current_chain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "type": "call",
                "expiration": EXPIRY,
                "daysToExpiration": 30,
                "strike": 100.0,
                "computedIV": 0.23,
            },
            {
                "type": "put",
                "expiration": EXPIRY,
                "daysToExpiration": 30,
                "strike": 100.0,
                "computedIV": 0.21,
            },
            {
                "type": "call",
                "expiration": EXPIRY,
                "daysToExpiration": 30,
                "strike": 110.0,
                "computedIV": 0.29,
            },
        ]
    )


def test_surface_change_matches_latest_prior_snapshot(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 1, 10), (0.18, 0.19, 0.21)), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 2, 10), (0.20, 0.22, 0.25)), tmp_path)

    result = surface_change_analytics(
        "AAPL",
        _current_chain(),
        100.0,
        tmp_path,
        current_timestamp=datetime(2026, 5, 3, 10),
    )

    assert result["available"]
    assert result["previous_snapshot_timestamp"] == "2026-05-02T10:00:00"
    assert result["matched_points"] == 3
    assert result["mean_iv_change"] == pytest.approx(0.02)
    assert result["atm_change"]["iv_change"] == pytest.approx(0.03)
    assert result["expiry_changes"][0]["current_median_iv"] == pytest.approx(0.23)
    assert result["expiry_changes"][0]["previous_median_iv"] == pytest.approx(0.22)
    assert result["top_changes"][0]["iv_change"] == pytest.approx(0.04)


def test_surface_change_skips_snapshots_at_or_after_current_timestamp(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 2, 10), (0.20, 0.22, 0.25)), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 3, 10), (0.40, 0.42, 0.45)), tmp_path)

    result = surface_change_analytics(
        "AAPL",
        _current_chain(),
        100.0,
        tmp_path,
        current_timestamp=datetime(2026, 5, 3, 10),
    )

    assert result["available"]
    assert result["previous_snapshot_timestamp"] == "2026-05-02T10:00:00"


def test_atm_iv_vol_of_vol_uses_snapshot_changes_plus_current(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 1, 10), (0.20, 0.20, 0.25)), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 2, 10), (0.22, 0.22, 0.25)), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 3, 10), (0.24, 0.24, 0.25)), tmp_path)

    result = atm_iv_vol_of_vol_from_snapshots(
        "AAPL",
        tmp_path,
        current_iv=0.23,
        current_timestamp=datetime(2026, 5, 4, 10),
    )

    assert result["available"]
    assert result["observations"] == 4
    assert result["change_observations"] == 3
    assert result["mean_abs_change"] == pytest.approx((0.02 + 0.02 + 0.01) / 3)
    assert result["snapshot_vol_of_vol"] > 0
    assert result["annualized_vol_of_vol"] > result["snapshot_vol_of_vol"]


def test_connector_surface_change_metadata_flattens_dashboard_fields(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 2, 10), (0.20, 0.22, 0.25)), tmp_path)
    connector = DashboardConnector.__new__(DashboardConnector)
    connector.snapshot_dir = tmp_path

    metadata = connector._surface_change_metadata(
        "AAPL",
        _current_chain(),
        100.0,
        {"timestamp": datetime(2026, 5, 3, 10)},
    )

    assert metadata["surface_change_available"] is True
    assert metadata["surface_change_points"] == 3
    assert metadata["atm_iv_change"] == pytest.approx(0.03)
    assert metadata["surface_change"]["available"] is True
