import math

import numpy as np
import pandas as pd
import pytest

from src.dashboard.surface_view import surface_axis, surface_stats
from src.quant.forwards import apply_forward_metrics, discount_factor, expiry_forward_metadata, forward_price
from src.quant.skew import delta_skew_by_expiry, term_structure_metrics


def test_forward_price_discount_and_moneyness_metrics_are_deterministic():
    frame = pd.DataFrame(
        [
            {
                "expiration": pd.Timestamp("2026-06-19"),
                "daysToExpiration": 45,
                "strike": 105.0,
                "riskFreeRate": 0.05,
                "effectiveDividendYield": 0.02,
            }
        ]
    )

    enriched = apply_forward_metrics(frame, spot=100.0)
    expected_forward = 100.0 * math.exp((0.05 - 0.02) * 45 / 365)

    assert discount_factor(0.05, 45) == pytest.approx(math.exp(-0.05 * 45 / 365))
    assert forward_price(100.0, 45, 0.05, 0.02) == pytest.approx(expected_forward)
    assert enriched.iloc[0]["forwardPrice"] == pytest.approx(expected_forward)
    assert enriched.iloc[0]["discountFactor"] == pytest.approx(math.exp(-0.05 * 45 / 365))
    assert enriched.iloc[0]["forwardMoneyness"] == pytest.approx(105.0 / expected_forward)
    assert enriched.iloc[0]["logMoneyness"] == pytest.approx(math.log(105.0 / expected_forward))
    assert expiry_forward_metadata(enriched)["2026-06-19"]["forward_price"] == pytest.approx(expected_forward)


def test_surface_axis_supports_moneyness_log_moneyness_and_delta():
    strikes = np.array([90.0, 100.0, 110.0])
    expiries = np.array([30.0, 60.0])
    vols = np.array([[0.25, 0.22, 0.24], [0.27, 0.23, 0.25]])

    money, _, _, money_title, _, _ = surface_axis(strikes, expiries, vols, 100.0, "Moneyness")
    log_money, _, _, log_title, _, _ = surface_axis(strikes, expiries, vols, 100.0, "Log-moneyness")
    delta, _, _, delta_title, _, _ = surface_axis(strikes, expiries, vols, 100.0, "Call delta")

    assert money_title == "Moneyness (K/S)"
    assert log_title == "Log-moneyness ln(K/S)"
    assert delta_title == "Call delta"
    assert money[0].tolist() == pytest.approx([0.9, 1.0, 1.1])
    assert log_money[0].tolist() == pytest.approx([math.log(0.9), 0.0, math.log(1.1)])
    assert delta[0, 0] > delta[0, 1] > delta[0, 2]


def test_delta_skew_metrics_use_computed_iv_by_expiry():
    expiry = pd.Timestamp("2026-06-19")
    chain = pd.DataFrame(
        [
            {"type": "put", "expiration": expiry, "daysToExpiration": 45, "strike": 80, "computedIV": 0.35},
            {"type": "put", "expiration": expiry, "daysToExpiration": 45, "strike": 90, "computedIV": 0.30},
            {"type": "put", "expiration": expiry, "daysToExpiration": 45, "strike": 100, "computedIV": 0.25},
            {"type": "call", "expiration": expiry, "daysToExpiration": 45, "strike": 100, "computedIV": 0.24},
            {"type": "call", "expiration": expiry, "daysToExpiration": 45, "strike": 110, "computedIV": 0.22},
            {"type": "call", "expiration": expiry, "daysToExpiration": 45, "strike": 120, "computedIV": 0.21},
        ]
    )

    metrics = delta_skew_by_expiry(chain, spot=100.0)
    row = metrics.iloc[0]

    assert row["expiration"] == "2026-06-19"
    assert row["25d_put_iv"] >= row["25d_call_iv"]
    assert row["risk_reversal_25d"] == pytest.approx(row["25d_call_iv"] - row["25d_put_iv"])
    assert row["butterfly_25d"] == pytest.approx((row["25d_call_iv"] + row["25d_put_iv"]) / 2 - row["atm_iv"])


def test_term_structure_metrics_identify_regime_slope_and_curvature():
    metrics = term_structure_metrics([(30, 0.20), (60, 0.24), (120, 0.30)])

    assert metrics["regime"] == "contango"
    assert metrics["front_back_spread"] == pytest.approx(0.10)
    assert metrics["slope_per_30d"] == pytest.approx(0.10 / 90 * 30)
    assert metrics["curvature"] is not None
    assert surface_stats([90, 100, 110], [30, 60], [[0.21, 0.20, 0.22], [0.25, 0.24, 0.26]], 100)[
        "term_metrics"
    ]["regime"] == "contango"
