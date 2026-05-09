from datetime import datetime, timedelta

import numpy as np
import pytest

from src.data.models import MarketDataSnapshot, OptionQuote
from src.data.snapshots import save_snapshot
from src.quant.surface_prior import blend_surface_with_prior, load_historical_surface_prior


EXPIRIES = (
    (datetime(2026, 6, 19), 30),
    (datetime(2026, 7, 17), 60),
)


def _snapshot(timestamp: datetime, bump: float = 0.0) -> MarketDataSnapshot:
    options = []
    for expiry, dte in EXPIRIES:
        for strike, base_iv in ((90.0, 0.24), (100.0, 0.20), (110.0, 0.23)):
            log_money = float(np.log(strike / 100.0))
            options.append(
                OptionQuote(
                    contract=f"AAPL{timestamp:%Y%m%d}{expiry:%m%d}{int(strike)}",
                    type="call",
                    strike=strike,
                    expiry=expiry,
                    dte=dte,
                    raw_iv=base_iv + bump,
                    computed_iv=base_iv + bump,
                    log_moneyness=log_money,
                )
            )
    return MarketDataSnapshot(
        symbol="AAPL",
        spot=100.0,
        spot_timestamp=timestamp,
        chain_timestamp=timestamp,
        expirations=tuple(expiry for expiry, _ in EXPIRIES),
        options=tuple(options),
        source="fixture",
        mode="Stored",
        raw_rows=len(options),
        valid_rows=len(options),
    )


def test_historical_surface_prior_loads_deterministic_grid_from_snapshots(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 1, 10), bump=0.00), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 2, 10), bump=0.02), tmp_path)

    prior = load_historical_surface_prior(
        "AAPL",
        tmp_path,
        as_of=datetime(2026, 5, 3, 10),
        max_age=timedelta(days=4),
        log_moneyness_step=0.05,
        dte_step=30,
    )

    assert prior.available
    assert prior.snapshot_count == 2
    assert prior.source_point_count == 12
    assert prior.cell_count == 6
    assert prior.latest_snapshot_timestamp == "2026-05-02T10:00:00"
    assert prior.latest_age_days == pytest.approx(1.0)
    assert prior.metadata()["provenance"] == "historical_prior_estimate_not_market_observation"

    records = prior.records()
    assert [row["dte"] for row in records] == [30.0, 30.0, 30.0, 60.0, 60.0, 60.0]
    atm_30 = next(row for row in records if row["dte"] == 30.0 and row["log_moneyness"] == 0.0)
    assert atm_30["prior_iv"] == pytest.approx(0.21)
    assert atm_30["prior_iv_mean"] == pytest.approx(0.21)
    assert atm_30["observations"] == 2
    assert atm_30["snapshot_count"] == 2
    assert atm_30["provenance"] == "historical_prior_estimate_not_market_observation"


def test_historical_surface_prior_refuses_stale_history(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 1, 10)), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 2, 10)), tmp_path)

    prior = load_historical_surface_prior(
        "AAPL",
        tmp_path,
        as_of=datetime(2026, 5, 10, 10),
        max_age=timedelta(days=3),
    )

    assert not prior.available
    assert prior.reason == "Latest persisted snapshot is stale"
    assert prior.latest_snapshot_timestamp == "2026-05-02T10:00:00"
    assert prior.latest_age_days == pytest.approx(8.0)
    assert prior.grid.empty


def test_historical_surface_prior_refuses_insufficient_history(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 2, 10)), tmp_path)

    prior = load_historical_surface_prior(
        "AAPL",
        tmp_path,
        as_of=datetime(2026, 5, 3, 10),
        max_age=timedelta(days=3),
    )

    assert not prior.available
    assert prior.reason == "Need at least 2 recent snapshots with usable IV rows"
    assert prior.snapshot_count == 1
    assert prior.cell_count == 0


def test_prior_blending_uses_quality_recency_and_overlap(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 1, 10), bump=0.00), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 2, 10), bump=0.02), tmp_path)
    prior = load_historical_surface_prior(
        "AAPL",
        tmp_path,
        as_of=datetime(2026, 5, 3, 10),
        max_age=timedelta(days=4),
        log_moneyness_step=0.05,
        dte_step=30,
    )
    strikes = 100.0 * np.exp(np.array([-0.10, 0.0, 0.10]))
    expiries = np.array([30.0, 60.0])
    vols = _prior_vol_matrix(prior.records(), expiries, [-0.10, 0.0, 0.10]) + 0.02

    blended, meta = blend_surface_with_prior(
        strikes,
        expiries,
        vols,
        100.0,
        prior,
        quality_score=35.0,
        min_quality_score=70.0,
        max_blend_weight=0.40,
    )

    assert meta["applied"]
    assert meta["prior_source"] == "persisted_snapshots"
    assert meta["prior_age_days"] == pytest.approx(1.0)
    assert meta["overlap_count"] == 6
    assert meta["blend_weight"] == pytest.approx(0.15)
    assert meta["provenance"] == "historical_prior_estimate_not_market_observation"
    assert blended[0, 1] == pytest.approx((0.85 * 0.23) + (0.15 * 0.21))

    unblended, high_quality_meta = blend_surface_with_prior(
        strikes,
        expiries,
        vols,
        100.0,
        prior,
        quality_score=85.0,
    )
    assert not high_quality_meta["applied"]
    assert high_quality_meta["reason"] == "Current quality score is adequate"
    assert np.array_equal(unblended, vols)


def test_prior_blending_stabilizes_single_spike_but_not_broad_jump(tmp_path):
    save_snapshot(_snapshot(datetime(2026, 5, 1, 10), bump=0.00), tmp_path)
    save_snapshot(_snapshot(datetime(2026, 5, 2, 10), bump=0.02), tmp_path)
    prior = load_historical_surface_prior(
        "AAPL",
        tmp_path,
        as_of=datetime(2026, 5, 3, 10),
        max_age=timedelta(days=4),
        log_moneyness_step=0.05,
        dte_step=30,
    )
    strikes = 100.0 * np.exp(np.array([-0.10, 0.0, 0.10]))
    expiries = np.array([30.0, 60.0])
    prior_vols = _prior_vol_matrix(prior.records(), expiries, [-0.10, 0.0, 0.10])

    spike_vols = prior_vols.copy()
    spike_vols[0, 1] = 0.60
    spike_blended, spike_meta = blend_surface_with_prior(
        strikes,
        expiries,
        spike_vols,
        100.0,
        prior,
        quality_score=35.0,
        max_blend_weight=0.40,
    )

    assert spike_meta["applied"]
    assert not spike_meta["jump_detection"]["broad_shift_detected"]
    assert spike_blended[0, 1] < spike_vols[0, 1]

    broad_jump_vols = prior_vols + 0.08
    broad_blended, broad_meta = blend_surface_with_prior(
        strikes,
        expiries,
        broad_jump_vols,
        100.0,
        prior,
        quality_score=35.0,
        max_blend_weight=0.40,
    )

    assert not broad_meta["applied"]
    assert broad_meta["reason"] == "Current clean quotes indicate broad IV shift"
    assert broad_meta["jump_detection"]["broad_shift_detected"]
    assert broad_meta["jump_detection"]["median_change"] == pytest.approx(0.08)
    assert np.array_equal(broad_blended, broad_jump_vols)


def _prior_vol_matrix(records, expiries, log_moneyness):
    by_key = {(row["dte"], round(row["log_moneyness"], 2)): row["prior_iv"] for row in records}
    return np.array(
        [
            [by_key[(float(dte), round(float(log_money), 2))] for log_money in log_moneyness]
            for dte in expiries
        ]
    )
