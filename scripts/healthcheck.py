"""Project healthcheck for CI and local smoke testing."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _run(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        detail = fn()
        return CheckResult(name, True, detail)
    except Exception as exc:
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}")


def check_imports() -> str:
    import dashboard_connector  # noqa: F401
    from src.dashboard import run_dashboard  # noqa: F401
    from src.dashboard.loading import LoadingState, render_loading_state  # noqa: F401
    from src.dashboard.surface_view import surface_stats  # noqa: F401
    from src.dashboard.tables import filter_option_chain  # noqa: F401
    from src.dashboard.theme import apply_chart_layout  # noqa: F401
    from src.dashboard.tooltips import COLUMN_HELP, CONTROL_HELP  # noqa: F401
    from src.data.historical import HistoricalPriceLoader  # noqa: F401
    from src.data.market_calendar import MarketCalendar  # noqa: F401
    from src.data.models import MarketDataSnapshot, OptionQuote  # noqa: F401
    from src.data.options_provider import YFinanceOptionsProvider  # noqa: F401
    from src.data.retry import call_with_backoff  # noqa: F401
    from src.data.snapshots import save_snapshot  # noqa: F401
    from src.pricing.black_scholes import BlackScholesModel  # noqa: F401
    from src.pricing.implied_vol import ImpliedVolatilityCalculator  # noqa: F401
    from src.quant.arbitrage import apply_no_arbitrage_checks  # noqa: F401
    from src.quant.corporate_actions import CorporateActionProvider  # noqa: F401
    from src.quant.dividends import DividendProvider  # noqa: F401
    from src.quant.rates import RiskFreeRateProvider  # noqa: F401
    from src.quant.smoothing import smooth_iv_surface  # noqa: F401
    from src.quant.svi import calibrate_svi_by_expiry  # noqa: F401

    return "core and dashboard modules imported"


def check_pricing() -> str:
    from src.pricing.black_scholes import BlackScholesModel
    from src.pricing.implied_vol import ImpliedVolatilityCalculator
    from src.quant.dividends import DividendProvider
    from src.quant.rates import RiskFreeRateProvider

    rate = RiskFreeRateProvider().get_curve().rate_for_dte(180).rate
    dividend_yield = DividendProvider().get("AAPL").annual_yield
    price = BlackScholesModel.call_price(100.0, 100.0, 0.5, rate, 0.25, q=dividend_yield)
    iv, method = ImpliedVolatilityCalculator().calculate_implied_vol(
        price, 100.0, 100.0, 0.5, rate, "call", q=dividend_yield, method="brent"
    )
    if iv is None or abs(iv - 0.25) > 1e-4:
        raise AssertionError(f"IV round trip failed: {iv}")
    return f"call={price:.4f}, iv={iv:.4f}, method={method}"


def check_surface() -> str:
    from src.analysis.surface_builder import build_surface
    from src.data.price_provider import RealTimePriceProvider
    from src.data.synthetic_options import SyntheticOptionsGenerator

    provider = RealTimePriceProvider()
    generator = SyntheticOptionsGenerator(provider)
    chain = generator.create_chain("AAPL")
    spot = provider.get_live_price("AAPL")
    _, _, vols = build_surface(chain, spot, "AAPL")
    if vols.size == 0:
        raise AssertionError("surface has no points")
    return f"surface shape={vols.shape}, rows={len(chain)}"


def check_connector() -> str:
    from dashboard_connector import DashboardConnector

    connector = DashboardConnector()
    data = connector.get_current_data("AAPL")
    required = {"price", "data_mode", "iv_30d", "timestamp"}
    missing = required - set(data)
    if missing:
        raise AssertionError(f"missing connector fields: {sorted(missing)}")
    health = connector.get_system_health()
    snapshot = connector.get_market_data_snapshot("AAPL")
    if snapshot.symbol != "AAPL":
        raise AssertionError("snapshot symbol mismatch")
    market_status = connector.get_market_status()
    return (
        f"mode={data['data_mode']}, yfinance={health['overall'].get('yfinance_available')}, "
        f"snapshot_options={len(snapshot.options)}, market={market_status.get('session_state')}"
    )


def check_streamlit_testing() -> str:
    from streamlit.testing.v1 import AppTest

    previous_test_env = os.environ.get("PYTEST_CURRENT_TEST")
    os.environ["PYTEST_CURRENT_TEST"] = "scripts.healthcheck::streamlit_offline_smoke"
    at = AppTest.from_file("app.py")
    try:
        at.run(timeout=90)
    finally:
        if previous_test_env is None:
            os.environ.pop("PYTEST_CURRENT_TEST", None)
        else:
            os.environ["PYTEST_CURRENT_TEST"] = previous_test_env
    if len(at.exception) > 0:
        raise AssertionError(f"{len(at.exception)} Streamlit exceptions")
    return f"metrics={len(at.metric)}, dataframes={len(at.dataframe)}"


def main() -> int:
    checks: List[CheckResult] = [
        _run("imports", check_imports),
        _run("pricing", check_pricing),
        _run("surface", check_surface),
        _run("connector", check_connector),
        _run("streamlit", check_streamlit_testing),
    ]

    print("PROJECT HEALTHCHECK")
    print("=" * 60)
    for result in checks:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status:4} {result.name:12} {result.detail}")

    failed = [result for result in checks if not result.passed]
    if failed:
        print("=" * 60)
        print(f"{len(failed)} check(s) failed")
        return 1
    print("=" * 60)
    print("All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
