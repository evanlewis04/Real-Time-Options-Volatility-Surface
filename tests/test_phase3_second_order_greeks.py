import pytest

from diagnostic import debug_greeks
from src.pricing.black_scholes import BlackScholesModel, OptionGreeks


CASE = dict(S=105.0, K=100.0, T=45.0 / 365.0, r=0.042, sigma=0.31, q=0.012)


def test_second_order_greeks_match_bumped_bsm_sensitivities():
    call = "call"
    vol_step = 0.002
    spot_step = 0.50
    day = 1.0 / 365.0

    vanna_bump = (
        OptionGreeks.delta(option_type=call, **{**CASE, "sigma": CASE["sigma"] + vol_step})
        - OptionGreeks.delta(option_type=call, **{**CASE, "sigma": CASE["sigma"] - vol_step})
    ) / (2.0 * vol_step) * 0.01
    volga_bump = (
        OptionGreeks.vega(**{**CASE, "sigma": CASE["sigma"] + vol_step})
        - OptionGreeks.vega(**{**CASE, "sigma": CASE["sigma"] - vol_step})
    ) / (2.0 * vol_step) * 0.01
    speed_bump = (
        OptionGreeks.gamma(**{**CASE, "S": CASE["S"] + spot_step})
        - OptionGreeks.gamma(**{**CASE, "S": CASE["S"] - spot_step})
    ) / (2.0 * spot_step)
    charm_bump = (
        OptionGreeks.delta(option_type=call, **{**CASE, "T": CASE["T"] - day})
        - OptionGreeks.delta(option_type=call, **CASE)
    )
    color_bump = OptionGreeks.gamma(**{**CASE, "T": CASE["T"] - day}) - OptionGreeks.gamma(**CASE)

    assert OptionGreeks.vanna(**CASE) == pytest.approx(vanna_bump, abs=2e-6)
    assert OptionGreeks.volga(**CASE) == pytest.approx(volga_bump, abs=2e-6)
    assert OptionGreeks.vomma(**CASE) == pytest.approx(OptionGreeks.volga(**CASE), abs=1e-12)
    assert OptionGreeks.charm(option_type=call, **CASE) == pytest.approx(charm_bump, abs=1e-12)
    assert OptionGreeks.speed(**CASE) == pytest.approx(speed_bump, abs=2e-6)
    assert OptionGreeks.color(**CASE) == pytest.approx(color_bump, abs=1e-12)


def test_first_order_greek_units_match_project_conventions():
    call = "call"
    base = BlackScholesModel.call_price(**CASE)
    one_vol_point = BlackScholesModel.call_price(**{**CASE, "sigma": CASE["sigma"] + 0.01})
    one_rate_point = BlackScholesModel.call_price(**{**CASE, "r": CASE["r"] + 0.01})
    one_day_less = BlackScholesModel.call_price(**{**CASE, "T": CASE["T"] - 1.0 / 365.0})

    assert OptionGreeks.vega(**CASE) == pytest.approx(one_vol_point - base, abs=8e-4)
    assert OptionGreeks.rho(option_type=call, **CASE) == pytest.approx(one_rate_point - base, abs=6e-4)
    assert OptionGreeks.theta(option_type=call, **CASE) == pytest.approx(one_day_less - base, abs=8e-4)


def test_pltr_diagnostic_accepts_vega_per_one_volatility_point(capsys):
    capsys.readouterr()
    debug_greeks()
    output = capsys.readouterr().out

    assert "Vega/1%" in output
    assert "seems too low" not in output
