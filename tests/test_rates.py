from datetime import datetime

import pandas as pd
import pytest

from src.quant.rates import (
    LocalCurveRateSource,
    RateCurve,
    RatePoint,
    RiskFreeRateProvider,
    apply_curve_to_options,
    expiry_rate_metadata,
)


def test_local_curve_source_interpolates_configured_rates(tmp_path):
    curve_path = tmp_path / "rates.csv"
    curve_path.write_text("tenor_days,rate,label\n30,0.04,1m\n90,0.05,3m\n", encoding="utf-8")

    curve = LocalCurveRateSource(curve_path).load_curve()
    lookup = curve.rate_for_dte(60)

    assert curve.mode == "Local"
    assert lookup.rate == pytest.approx(0.045)
    assert curve.discount_factor(60) == pytest.approx(0.99263, abs=1e-5)


def test_missing_local_curve_uses_built_in_offline_fallback(tmp_path):
    curve = LocalCurveRateSource(tmp_path / "missing.csv").load_curve()

    assert curve.mode == "Fallback"
    assert curve.fallback_reason
    assert curve.rate_for_dte(30).rate > 0.0


def test_provider_falls_back_to_local_when_live_source_is_unavailable(tmp_path):
    curve_path = tmp_path / "rates.csv"
    curve_path.write_text("tenor_days,rate\n30,0.03\n365,0.04\n", encoding="utf-8")

    provider = RiskFreeRateProvider(preferred_source="fred", local_curve_path=curve_path)
    curve = provider.get_curve()

    assert curve.mode == "Fallback"
    assert "FRED" in curve.fallback_reason
    assert curve.rate_for_dte(30).rate == pytest.approx(0.03)


def test_apply_curve_to_options_adds_expiry_specific_rates():
    curve = RateCurve(
        as_of=datetime(2026, 5, 3),
        source="fixture",
        mode="Local",
        points=(RatePoint(30, 0.04), RatePoint(90, 0.05)),
    )
    frame = pd.DataFrame(
        [
            {"expiration": pd.Timestamp("2026-06-02"), "daysToExpiration": 30},
            {"expiration": pd.Timestamp("2026-08-01"), "daysToExpiration": 90},
        ]
    )

    enriched = apply_curve_to_options(frame, curve)

    assert enriched["riskFreeRate"].tolist() == pytest.approx([0.04, 0.05])
    assert expiry_rate_metadata(enriched, curve) == {"2026-06-02": 0.04, "2026-08-01": 0.05}
