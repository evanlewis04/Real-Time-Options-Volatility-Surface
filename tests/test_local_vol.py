import numpy as np

from src.quant.local_vol import dupire_local_vol_surface


def test_dupire_local_vol_surface_enables_for_smoothed_quality_grid():
    strikes = np.array([80.0, 100.0, 120.0])
    expiries = np.array([30.0, 60.0, 90.0])
    vols = np.full((3, 3), 0.25)

    result = dupire_local_vol_surface(
        strikes,
        expiries,
        vols,
        spot=100.0,
        quality_score=95.0,
        smoothing_meta={"method": "gaussian_blend_calendar_total_variance"},
    )

    assert result["enabled"]
    assert result["method"] == "dupire_total_variance_log_moneyness"
    assert result["invalid_points"] == 0
    assert np.nanmean(np.asarray(result["grid"], dtype=float)) > 0.20


def test_dupire_local_vol_surface_disables_when_quality_is_low():
    result = dupire_local_vol_surface(
        np.array([80.0, 100.0, 120.0]),
        np.array([30.0, 60.0, 90.0]),
        np.full((3, 3), 0.25),
        spot=100.0,
        quality_score=40.0,
        smoothing_meta={"method": "gaussian_blend_calendar_total_variance"},
    )

    assert not result["enabled"]
    assert "quality" in result["reason"].lower()
