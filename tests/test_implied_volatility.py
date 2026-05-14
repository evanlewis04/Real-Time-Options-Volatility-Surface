import pytest

from src.pricing.black_scholes import BlackScholesModel
from src.pricing.implied_vol import ImpliedVolatilityCalculator


@pytest.fixture
def iv_calculator():
    return ImpliedVolatilityCalculator(max_iterations=100, tolerance=1e-8)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.35, 0.65])
def test_newton_raphson_round_trip(iv_calculator, option_type, sigma_true):
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.04, 0.0
    market_price = BlackScholesModel.option_price(S, K, T, r, sigma_true, option_type, q)

    iv = iv_calculator.newton_raphson(market_price, S, K, T, r, option_type, q)

    assert iv is not None
    assert iv == pytest.approx(sigma_true, abs=1e-4)


@pytest.mark.parametrize("moneyness", [0.8, 1.0, 1.2])
def test_round_trip_holds_across_moneyness(iv_calculator, moneyness):
    S, T, r, sigma_true = 100.0, 0.25, 0.03, 0.30
    K = S * moneyness
    market_price = BlackScholesModel.call_price(S, K, T, r, sigma_true)

    iv = iv_calculator.newton_raphson(market_price, S, K, T, r, "call")

    assert iv is not None
    assert iv == pytest.approx(sigma_true, abs=1e-4)


def test_arbitrage_violation_returns_none(iv_calculator):
    # A call worth less than its intrinsic value is an arbitrage and has no IV
    S, K, T, r = 100.0, 80.0, 0.5, 0.05
    intrinsic_lower_bound = S - K  # ignoring discount
    bad_price = max(0.0, intrinsic_lower_bound) - 5.0

    with pytest.warns(UserWarning, match="below intrinsic value"):
        iv = iv_calculator.newton_raphson(bad_price, S, K, T, r, "call")

    assert iv is None


def test_invalid_inputs_return_none(iv_calculator):
    with pytest.warns(UserWarning, match="Market price must be positive"):
        assert iv_calculator.newton_raphson(-1.0, 100.0, 100.0, 0.5, 0.05, "call") is None
    with pytest.warns(UserWarning, match="Stock price must be positive"):
        assert iv_calculator.newton_raphson(5.0, 0.0, 100.0, 0.5, 0.05, "call") is None
    with pytest.warns(UserWarning, match="Time to expiration cannot be negative"):
        assert iv_calculator.newton_raphson(5.0, 100.0, 100.0, -0.1, 0.05, "call") is None
