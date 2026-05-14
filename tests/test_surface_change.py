from datetime import datetime

import pandas as pd
import pytest

from dashboard_connector import DashboardConnector
from src.data.models import MarketDataSnapshot, OptionQuote
from src.data.snapshots import save_snapshot
from src.quant.surface_change import (
    atm_iv_vol_of_vol_from_snapshots,
    rich_cheap_scanner,
    surface_shape_change_quality_flag,
    surface_change_analytics,
    surface_change_heatmaps,
    surface_tape_analytics,
)


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


def _scanner_chain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contractSymbol": "AAPL260619C00090000",
                "type": "call",
                "expiration": EXPIRY,
                "daysToExpiration": 30,
                "strike": 90.0,
                "computedIV": 0.24,
                "bidAskSpreadPct": 0.08,
                "volume": 20,
                "openInterest": 150,
                "quoteReliabilityScore": 0.90,
            },
            {
                "contractSymbol": "AAPL260619C00100000",
                "type": "call",
                "expiration": EXPIRY,
                "daysToExpiration": 30,
                "strike": 100.0,
                "computedIV": 0.31,
                "bidAskSpreadPct": 0.04,
                "volume": 120,
                "openInterest": 800,
                "quoteReliabilityScore": 0.10,
            },
            {
                "contractSymbol": "AAPL260619P00110000",
                "type": "put",
                "expiration": EXPIRY,
                "daysToExpiration": 30,
                "strike": 110.0,
                "computedIV": 0.19,
                "bidAskSpreadPct": 0.10,
                "volume": 40,
                "openInterest": 250,
                "quoteReliabilityScore": 0.95,
            },
        ]
    )


def _svi_smiles() -> list[dict]:
    return [
        {
            "expiration": "2026-06-19",
            "residuals": [
                {"strike": 90.0, "observed_iv": 0.24, "fitted_iv": 0.23, "residual": -0.01},
                {"strike": 100.0, "observed_iv": 0.31, "fitted_iv": 0.25, "residual": -0.06},
                {"strike": 110.0, "observed_iv": 0.19, "fitted_iv": 0.24, "residual": 0.05},
            ],
        }
    ]


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
    assert result["heatmaps"]["available"] is True
    assert result["heatmaps"]["baselines"]["previous_refresh"]["records"][0]["iv_change"] == pytest.approx(0.01)
    assert result["tape"]["snapshot_count"] == 1


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


def test_surface_change_heatmaps_select_refresh_hour_and_close_baselines(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 2, 15, 55), (0.19, 0.20, 0.23)), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 3, 8, 30), (0.20, 0.21, 0.24)), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 3, 9, 45), (0.21, 0.22, 0.25)), tmp_path)

    result = surface_change_heatmaps(
        "AAPL",
        _current_chain(),
        tmp_path,
        current_timestamp=datetime(2026, 5, 3, 10, 50),
    )

    baselines = result["baselines"]
    assert result["available"] is True
    assert baselines["previous_refresh"]["baseline_timestamp"] == "2026-05-03T09:45:00"
    assert baselines["previous_hour"]["baseline_timestamp"] == "2026-05-03T09:45:00"
    assert baselines["previous_close"]["baseline_timestamp"] == "2026-05-02T15:55:00"
    expected_pct = (((0.23 - 0.19) / 0.19) + ((0.21 - 0.20) / 0.20)) / 2
    assert baselines["previous_close"]["records"][0]["iv_change_pct"] == pytest.approx(expected_pct)


def test_surface_tape_returns_intraday_snapshots_plus_current(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 2, 15, 55), (0.19, 0.20, 0.23)), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 3, 9, 45), (0.21, 0.22, 0.25)), tmp_path)

    tape = surface_tape_analytics(
        "AAPL",
        tmp_path,
        current_chain=_current_chain(),
        current_timestamp=datetime(2026, 5, 3, 10, 50),
    )

    assert tape["available"] is True
    assert tape["snapshot_count"] == 2
    assert tape["timestamps"] == ["2026-05-03T09:45:00", "2026-05-03T10:50:00"]
    assert tape["snapshots"][-1]["mode"] == "Current"
    assert tape["snapshots"][-1]["points"][0]["iv"] == pytest.approx(0.22)


def test_rich_cheap_scanner_ranks_residuals_with_liquidity_reasons():
    scanner = rich_cheap_scanner(_scanner_chain(), _svi_smiles())

    assert scanner["available"] is True
    assert scanner["fit_mode"] == "Robust SVI"
    assert scanner["ranking_policy"] == "abs_residual_z_score_times_liquidity_and_quote_reliability"
    assert scanner["candidate_count"] == 3
    assert scanner["rich_count"] == 2
    assert scanner["cheap_count"] == 1
    top = scanner["candidates"][0]
    assert top["confidence_label"] in {"high", "medium"}
    assert top["quote_reliability_score"] >= 0.90
    assert "z-score" in top["reason"]
    low_confidence = next(row for row in scanner["candidates"] if row["contract"] == "AAPL260619C00100000")
    assert low_confidence["confidence_label"] == "low"


def test_surface_shape_change_flags_quality_driven_move():
    flag = surface_shape_change_quality_flag(
        {
            "available": True,
            "max_abs_iv_change": 0.08,
            "median_abs_iv_change": 0.03,
        },
        current_quality_score=84.0,
        previous_quality_score=98.0,
        current_reason_buckets={"no_arbitrage_violation": 6, "wide_bid_ask_spread": 2},
        previous_reason_buckets={"no_arbitrage_violation": 1},
    )

    assert flag["available"] is True
    assert flag["likely_data_quality_driven"] is True
    assert flag["deteriorated_buckets"]["no_arbitrage_violation"] == 5
    assert flag["provenance"] == "surface_change_quality_diagnostic_not_market_observation"


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
    assert metadata["surface_shape_change_quality"]["available"] is True
    assert metadata["surface_tape_available"] is True
    assert metadata["surface_change_heatmap_available"] is True


def test_connector_rich_cheap_metadata_flattens_scanner_fields():
    metadata = DashboardConnector._rich_cheap_metadata(_scanner_chain(), {"svi_smiles": _svi_smiles()})

    assert metadata["rich_cheap_scanner_available"] is True
    assert metadata["rich_cheap_candidates"] == 3
    assert metadata["rich_cheap_rich_count"] == 2
