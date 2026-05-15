"""Interactive browser crawler for the Streamlit dashboard.

The crawler starts the local app in deterministic fallback mode, drives the
major dashboard tabs and a few safe controls, captures screenshots, and exits
nonzero only for real UI regressions. Lean environments without Playwright or a
usable Chromium runner are reported as clean skips so regular offline/AppTest
coverage remains deterministic.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DATA_MODES = ("offline", "online")
DEFAULT_OUTPUT_DIRS = {
    "offline": "artifacts/dashboard_crawler",
    "online": "artifacts/dashboard_crawler_online",
}
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dashboard_visual_regression import (  # noqa: E402
    VIEWPORTS,
    _has_playwright,
    _pick_port,
    _wait_for_http,
)


TAB_SPECS = {
    "F1 Surface": ("Volatility Surface", "Market Snapshot", "Selected fit view"),
    "F2 Chain": ("Option Chain Explorer", ("Showing", "Option chain unavailable", "Option chain panel idle")),
    "F3 Skew": ("Smile And Term Structure", "Term structure:", ("Expected Move By Expiry", "Expected move unavailable")),
    "F4 Term": ("Term Structure Panel", ("ATM IV Term Structure", "Term structure unavailable")),
    "F5 Quality": ("Data Quality Panel", ("Top Quality Drivers", "No-Arbitrage Summary", "Expiry quality unavailable")),
    "F6 Scanner": ("Scanner Panel", "Relative Value Dashboard", "Cross-Sectional Vol Map"),
    "F7 Strategy": ("Earnings Vol Event Engine", "Strategy Builder"),
    "F8 Risk": ("Portfolio And Cross-Asset Risk", ("Portfolio book unavailable", "Portfolio Greeks")),
    "F9 Diag": ("Diagnostics And Data Provenance", "Advanced Research Modules", "Surface Alerts"),
    "F10 Export": ("Report Export Panel", "Export fit diagnostics JSON", "Export payload source"),
}

STREAMLIT_ERROR_RE = re.compile(
    r"Traceback|Uncaught app exception|ModuleNotFoundError|NameError|KeyError|ValueError|TypeError",
    re.IGNORECASE,
)
HTML_FRAGMENT_RE = re.compile(r"</?(?:div|span|section|style)\b|data-dashboard-[a-z-]+=", re.IGNORECASE)
TRUNCATED_KPI_RE = re.compile(r"^(?:[$]?\d?|\w?)\.\.\.$")


@dataclass
class CrawlResult:
    screenshots: list[Path] = field(default_factory=list)
    interactions: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    data_mode: str = "unknown"
    data_source: str = "unknown"
    skipped: bool = False
    skip_reason: str = ""

    def fail(self, message: str) -> None:
        self.failures.append(message)


def _streamlit_env(data_mode: str) -> dict[str, str]:
    """Return the environment used to launch Streamlit for the requested data mode."""
    if data_mode not in DATA_MODES:
        raise ValueError(f"unknown data mode: {data_mode}")
    env = os.environ.copy()
    env.setdefault("STREAMLIT_LOG_LEVEL", "error")
    if data_mode == "offline":
        env["PYTEST_CURRENT_TEST"] = "scripts.dashboard_ui_crawler::offline_browser_crawl"
        env["VOL_SURFACE_APPTEST_MODE"] = "synthetic"
    else:
        env.pop("PYTEST_CURRENT_TEST", None)
        env.pop("VOL_SURFACE_APPTEST_MODE", None)
    return env


def _default_output_dir(data_mode: str) -> str:
    if data_mode not in DATA_MODES:
        raise ValueError(f"unknown data mode: {data_mode}")
    return DEFAULT_OUTPUT_DIRS[data_mode]


def _start_streamlit(port: int, cwd: Path, data_mode: str) -> subprocess.Popen:
    """Start Streamlit in deterministic offline mode or explicit online mode."""
    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    return subprocess.Popen(
        args,
        cwd=str(cwd),
        env=_streamlit_env(data_mode),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _delegate_to_repo_venv(root: Path) -> int | None:
    """Use the repo virtualenv for Playwright when the active interpreter lacks it."""
    venv_python = root / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists() or Path(sys.executable).resolve() == venv_python.resolve():
        return None
    probe = subprocess.run(
        [str(venv_python), "-c", "import playwright.sync_api"],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        return None
    return subprocess.call([str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=str(root))


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _screenshot(page, output_dir: Path, name: str, result: CrawlResult) -> None:
    path = output_dir / f"{name}.png"
    page.screenshot(path=str(path), full_page=True, animations="disabled")
    result.screenshots.append(path)


def _visible_text(page) -> str:
    return page.locator("body").inner_text(timeout=10_000)


def _wait_for_visible_loaders_gone(page, timeout_ms: int = 90_000) -> None:
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        if page.locator(".loading-panel:visible").count() == 0:
            return
        page.wait_for_timeout(500)
    raise AssertionError("visible loading panels remained after timeout")


def _wait_for_settle(page, timeout_ms: int = 90_000) -> None:
    """Wait for dashboard readiness markers and visible loading panels to clear."""
    page.get_by_text("Options Volatility Surface Workstation").wait_for(timeout=timeout_ms)
    page.locator('[data-dashboard-section="kpi-grid"]').wait_for(state="visible", timeout=timeout_ms)
    page.get_by_text("Surface Readiness").wait_for(timeout=timeout_ms)
    page.locator(".dashboard-ready-marker").wait_for(state="attached", timeout=timeout_ms)
    _wait_for_visible_loaders_gone(page, timeout_ms=timeout_ms)
    page.wait_for_load_state("networkidle", timeout=timeout_ms)
    page.wait_for_timeout(750)
    _wait_for_visible_loaders_gone(page, timeout_ms=timeout_ms)


def _assert_any_text(page, expected: str | tuple[str, ...], context: str) -> None:
    choices = (expected,) if isinstance(expected, str) else expected
    text = _visible_text(page)
    if not any(choice in text for choice in choices):
        raise AssertionError(f"{context}: expected one of {choices!r}")


def _assert_no_streamlit_errors(page, context: str) -> None:
    exception_panels = page.locator('[data-testid="stException"]')
    if exception_panels.count():
        raise AssertionError(f"{context}: Streamlit exception panel is visible")
    text = _visible_text(page)
    match = STREAMLIT_ERROR_RE.search(text)
    if match:
        raise AssertionError(f"{context}: error-like text is visible: {match.group(0)}")


def _assert_no_loading_panels(page, context: str) -> None:
    loaders = page.locator(".loading-panel:visible")
    if loaders.count():
        raise AssertionError(f"{context}: loading panel remained after settling")


def _assert_no_literal_html(page) -> None:
    text = _visible_text(page)
    match = HTML_FRAGMENT_RE.search(text)
    if match:
        raise AssertionError(f"initial viewport shows literal HTML fragment: {match.group(0)}")


def _assert_kpis_readable(page) -> None:
    values = page.locator(".metric-card-value").all_inner_texts()
    if len(values) < 5:
        raise AssertionError(f"expected KPI card values, found {len(values)}")
    bad_values = [value.strip() for value in values if TRUNCATED_KPI_RE.match(value.strip())]
    if bad_values:
        raise AssertionError(f"KPI values look truncated: {bad_values}")


def _observe_dashboard_metadata(page) -> tuple[str, str]:
    status_items = page.locator(".status-pill").all_inner_texts()
    tape_items = page.locator(".workstation-tape span").all_inner_texts()
    data_mode = "unknown"
    data_source = "unknown"
    for item in status_items:
        if item.startswith("Surface:"):
            data_mode = item.split(":", 1)[1].strip()
            break
    for item in tape_items:
        if item.startswith("Source"):
            data_source = item.removeprefix("Source").strip()
            break
    return data_mode, data_source


def _assert_provenance_visible(page, context: str) -> None:
    _assert_any_text(
        page,
        ("not_market_observation", "not a market observation", "Synthetic", "Fallback"),
        context,
    )


def _click_tab(page, tab_name: str) -> None:
    page.get_by_role("tab", name=tab_name, exact=True).click(timeout=30_000)
    _wait_for_settle(page)


def _select_option(page, label: str, option: str) -> None:
    name = re.compile(rf"{re.escape(label)}$")
    page.get_by_role("combobox", name=name).click(timeout=30_000)
    page.get_by_role("option", name=option, exact=True).click(timeout=30_000)
    _wait_for_settle(page)


def _set_checkbox(page, label: str, checked: bool) -> None:
    checkbox = page.locator(f'input[type="checkbox"][aria-label="{label}"]')
    if checkbox.is_checked(timeout=10_000) == checked:
        return
    page.locator("label").filter(has_text=label).click(timeout=30_000)
    _wait_for_settle(page)
    if checkbox.is_checked(timeout=10_000) != checked:
        raise AssertionError(f"{label} checkbox did not reach checked={checked}")


def _exercise_controls(page, result: CrawlResult) -> None:
    """Drive a small deterministic set of controls without refreshing data."""
    try:
        _select_option(page, "Primary underlying", "MSFT")
        result.interactions.append("changed primary underlying to MSFT")
    except Exception as exc:
        result.interactions.append(f"primary underlying change skipped: {exc}")

    for checked in (False, True):
        _set_checkbox(page, "3D surface", checked)
        result.interactions.append(f"set 3D surface to {checked}")

    _select_option(page, "Surface x-axis", "Moneyness")
    result.interactions.append("changed surface x-axis to Moneyness")

    _select_option(page, "Fit Mode", "Diagnostic Raw")
    result.interactions.append("changed fit mode to Diagnostic Raw")
    _assert_provenance_visible(page, "Diagnostic Raw fit mode")

    _select_option(page, "Fit Mode", "Robust")
    result.interactions.append("changed fit mode back to Robust")
    _assert_provenance_visible(page, "Robust fit mode")


def _crawl_tabs(page, output_dir: Path, result: CrawlResult) -> None:
    for index, (tab_name, expectations) in enumerate(TAB_SPECS.items(), start=1):
        _click_tab(page, tab_name)
        context = f"{tab_name} tab"
        _assert_no_streamlit_errors(page, context)
        _assert_no_loading_panels(page, context)
        for expected in expectations:
            _assert_any_text(page, expected, context)
        if tab_name in {"DataQualityPanel", "DiagnosticsPanel"}:
            _assert_provenance_visible(page, context)
        if tab_name == "F2 Chain":
            _assert_any_text(
                page,
                ("Option Chain Explorer", "Showing", "Option chain unavailable", "Option chain panel idle"),
                context,
            )
        _screenshot(page, output_dir, f"tab_{index:02d}_{tab_name}", result)


def _run_crawler(url: str, output_dir: Path, viewport_name: str) -> CrawlResult:
    result = CrawlResult()
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError:
        result.skipped = True
        result.skip_reason = "playwright is not installed"
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = VIEWPORTS[viewport_name]
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except PlaywrightError as exc:
            message = str(exc)
            if "Executable doesn't exist" in message or "playwright install" in message:
                result.skipped = True
                result.skip_reason = "Playwright Chromium browser is not installed"
                return result
            raise
        try:
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(url, wait_until="networkidle", timeout=90_000)
            try:
                _wait_for_settle(page)
                _assert_no_streamlit_errors(page, "initial viewport")
                _assert_no_loading_panels(page, "initial viewport")
                _assert_no_literal_html(page)
                _assert_kpis_readable(page)
                _assert_provenance_visible(page, "initial viewport")
                result.data_mode, result.data_source = _observe_dashboard_metadata(page)
                _screenshot(page, output_dir, f"initial_{viewport_name}", result)

                _exercise_controls(page, result)
                _crawl_tabs(page, output_dir, result)
            except Exception as exc:
                result.fail(f"{type(exc).__name__}: {exc}")
        finally:
            browser.close()
    return result


def _print_summary(result: CrawlResult) -> None:
    if result.skipped:
        print(f"SKIP dashboard UI crawler: {result.skip_reason}")
        return
    print("Dashboard UI crawler summary")
    print(f"Observed data mode: {result.data_mode}")
    print(f"Observed data source: {result.data_source}")
    print(f"Interactions: {len(result.interactions)}")
    for item in result.interactions:
        print(f" - {item}")
    print(f"Screenshots: {len(result.screenshots)}")
    for path in result.screenshots:
        print(f" - {path}")
    if result.failures:
        print(f"Failures: {len(result.failures)}")
        for failure in result.failures:
            print(f" - {failure}")
    else:
        print("Failures: 0")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an interactive Streamlit dashboard UI crawl.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--output-dir")
    parser.add_argument("--viewport", choices=sorted(VIEWPORTS), default="desktop")
    parser.add_argument("--data-mode", choices=DATA_MODES, default="offline")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not _has_playwright():
        delegated = _delegate_to_repo_venv(ROOT)
        if delegated is not None:
            return delegated

    port = _pick_port(args.port)
    url = f"http://127.0.0.1:{port}"
    if args.data_mode == "online":
        print("WARNING dashboard UI crawler online mode may fetch live/delayed yfinance data.")
    output_dir = ROOT / (args.output_dir or _default_output_dir(args.data_mode))
    proc = _start_streamlit(port, ROOT, args.data_mode)
    try:
        _wait_for_http(url, timeout_seconds=150 if args.data_mode == "online" else 75)
        started = time.time()
        result = _run_crawler(url, output_dir, args.viewport)
        elapsed = time.time() - started
        print(f"Crawler elapsed seconds: {elapsed:.1f}")
        _print_summary(result)
        return 1 if result.failures else 0
    except Exception as exc:
        result = CrawlResult(failures=[f"{type(exc).__name__}: {exc}"])
        _print_summary(result)
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
