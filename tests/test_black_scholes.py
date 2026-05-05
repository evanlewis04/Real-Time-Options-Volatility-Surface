import math

import pytest

from src.pricing.black_scholes import BlackScholesModel, OptionGreeks


# Hull, "Options, Futures, and Other Derivatives" (9e), Example 15.6
# S=42, K=40, T=0.5, r=0.10, sigma=0.20, q=0
HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20, q=0.0)
HULL_CALL = 4.759
HULL_PUT = 0.8086
HULL_D1 = 0.7693
HULL_D2 = 0.6278


REFERENCE_CASES = [
    pytest.param(
        dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.03),
        8.652528553943,
        6.730917649163,
        id="continuous-dividend-atm",
    ),
    pytest.param(
        dict(S=100.0, K=95.0, T=2.0, r=0.08, sigma=0.35, q=0.0),
        28.756571280748,
        9.710231232538,
        id="higher-rate-two-year",
    ),
    pytest.param(
        dict(S=100.0, K=100.5, T=1.0 / 365.0, r=0.04, sigma=0.25, q=0.01),
        0.313785466815,
        0.805512060151,
        id="near-expiry-atm",
    ),
    pytest.param(
        dict(S=150.0, K=80.0, T=0.75, r=0.03, sigma=0.40, q=0.015),
        70.569109526547,
        0.467251810329,
        id="deep-itm-call",
    ),
    pytest.param(
        dict(S=50.0, K=120.0, T=1.25, r=0.02, sigma=0.50, q=0.0),
        1.160119076642,
        68.197308520042,
        id="deep-otm-call",
    ),
    pytest.param(
        dict(S=100.0, K=105.0, T=1.5, r=-0.01, sigma=0.30, q=0.025),
        10.205684376111,
        20.473114388679,
        id="negative-rate-with-dividend",
    ),
]


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _reference_prices(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    stock_pv = S * math.exp(-q * T)
    strike_pv = K * math.exp(-r * T)
    call = stock_pv * _normal_cdf(d1) - strike_pv * _normal_cdf(d2)
    put = strike_pv * _normal_cdf(-d2) - stock_pv * _normal_cdf(-d1)
    return call, put


def test_d1_d2_match_hull_example():
    assert BlackScholesModel.d1(**HULL) == pytest.approx(HULL_D1, abs=1e-3)
    assert BlackScholesModel.d2(**HULL) == pytest.approx(HULL_D2, abs=1e-3)


def test_call_price_matches_hull_example():
    assert BlackScholesModel.call_price(**HULL) == pytest.approx(HULL_CALL, abs=1e-2)


def test_put_price_matches_hull_example():
    assert BlackScholesModel.put_price(**HULL) == pytest.approx(HULL_PUT, abs=1e-2)


@pytest.mark.parametrize(("params", "expected_call", "expected_put"), REFERENCE_CASES)
def test_prices_match_independent_reference_cases(params, expected_call, expected_put):
    reference_call, reference_put = _reference_prices(**params)

    assert reference_call == pytest.approx(expected_call, abs=1e-12)
    assert reference_put == pytest.approx(expected_put, abs=1e-12)
    assert BlackScholesModel.call_price(**params) == pytest.approx(expected_call, abs=1e-10)
    assert BlackScholesModel.put_price(**params) == pytest.approx(expected_put, abs=1e-10)


def test_put_call_parity():
    call = BlackScholesModel.call_price(**HULL)
    put = BlackScholesModel.put_price(**HULL)
    assert BlackScholesModel.put_call_parity_check(
        call, put, HULL["S"], HULL["K"], HULL["T"], HULL["r"], HULL["q"]
    )


def test_put_call_parity_with_dividend():
    params = {**HULL, "q": 0.03}
    call = BlackScholesModel.call_price(**params)
    put = BlackScholesModel.put_price(**params)
    lhs = call - put
    rhs = params["S"] * math.exp(-params["q"] * params["T"]) - params["K"] * math.exp(-params["r"] * params["T"])
    assert lhs == pytest.approx(rhs, abs=1e-6)


@pytest.mark.parametrize(("params", "expected_call", "expected_put"), REFERENCE_CASES)
def test_put_call_parity_holds_across_reference_cases(params, expected_call, expected_put):
    lhs = expected_call - expected_put
    rhs = params["S"] * math.exp(-params["q"] * params["T"]) - params["K"] * math.exp(-params["r"] * params["T"])

    assert lhs == pytest.approx(rhs, abs=1e-10)
    assert BlackScholesModel.put_call_parity_check(
        expected_call,
        expected_put,
        params["S"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        tolerance=1e-10,
    )


def test_call_price_at_expiration_is_intrinsic():
    assert BlackScholesModel.call_price(50.0, 40.0, 0.0, 0.05, 0.2) == pytest.approx(10.0)
    assert BlackScholesModel.call_price(30.0, 40.0, 0.0, 0.05, 0.2) == pytest.approx(0.0)


def test_put_price_at_expiration_is_intrinsic():
    assert BlackScholesModel.put_price(30.0, 40.0, 0.0, 0.05, 0.2) == pytest.approx(10.0)
    assert BlackScholesModel.put_price(50.0, 40.0, 0.0, 0.05, 0.2) == pytest.approx(0.0)


def test_option_price_dispatch_matches_specialized_methods():
    call_via_dispatch = BlackScholesModel.option_price(option_type="call", **HULL)
    put_via_dispatch = BlackScholesModel.option_price(option_type="put", **HULL)
    assert call_via_dispatch == BlackScholesModel.call_price(**HULL)
    assert put_via_dispatch == BlackScholesModel.put_price(**HULL)


def test_option_price_rejects_unknown_type():
    with pytest.raises(ValueError):
        BlackScholesModel.option_price(option_type="straddle", **HULL)


def test_validate_inputs_rejects_non_positive_prices():
    with pytest.raises(ValueError):
        BlackScholesModel.validate_inputs(S=0.0, K=40.0, T=0.5, r=0.05, sigma=0.2)
    with pytest.raises(ValueError):
        BlackScholesModel.validate_inputs(S=42.0, K=-1.0, T=0.5, r=0.05, sigma=0.2)


def test_call_delta_in_unit_interval():
    delta = OptionGreeks.delta(option_type="call", **HULL)
    assert 0.0 < delta < 1.0


def test_put_delta_in_negative_unit_interval():
    delta = OptionGreeks.delta(option_type="put", **HULL)
    assert -1.0 < delta < 0.0


def test_call_put_delta_relation_no_dividend():
    # Without dividends: delta_call - delta_put = 1
    call_delta = OptionGreeks.delta(option_type="call", **HULL)
    put_delta = OptionGreeks.delta(option_type="put", **HULL)
    assert call_delta - put_delta == pytest.approx(1.0, abs=1e-9)


def test_gamma_is_positive():
    assert OptionGreeks.gamma(**HULL) > 0.0


def test_vega_is_positive():
    assert OptionGreeks.vega(**HULL) > 0.0


def test_theta_is_negative_for_long_options():
    assert OptionGreeks.theta(option_type="call", **HULL) < 0.0
    assert OptionGreeks.theta(option_type="put", **HULL) < 0.0


def test_zero_volatility_call_is_discounted_intrinsic():
    expected = 42.0 - 40.0 * math.exp(-0.10 * 0.5)
    assert BlackScholesModel.call_price(42.0, 40.0, 0.5, 0.10, 0.0) == pytest.approx(expected, abs=1e-9)
