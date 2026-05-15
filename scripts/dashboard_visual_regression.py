"""Optional visual-regression screenshot capture for the Streamlit dashboard.

The script starts the local Streamlit app, captures desktop/tablet/mobile
screenshots when Playwright is installed, and stores them under
``artifacts/dashboard_screenshots``. In lean environments without Playwright it
exits successfully with a clear skip message; structural AppTest coverage still
guards the dashboard in the regular test suite.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable


VIEWPORTS = {
    "desktop": (1440, 1000),
    "tablet": (1024, 900),
    "mobile": (390, 900),
}


def _port_open(port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _pick_port(preferred: int) -> int:
    port = preferred
    while _port_open(port):
        port += 1
    return port


def _wait_for_http(url: str, timeout_seconds: int = 75) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - diagnostic detail only
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Streamlit did not become ready at {url}: {last_error}")


def _start_streamlit(port: int, cwd: Path) -> subprocess.Popen:
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _has_playwright() -> bool:
    try:
        return importlib.util.find_spec("playwright.sync_api") is not None
    except ModuleNotFoundError:
        return False


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


def _wait_for_dashboard_settled(page) -> None:
    """Wait until Streamlit has rendered the workstation and cleared loaders."""
    page.get_by_text("Options Volatility Surface Workstation").wait_for(timeout=90_000)
    page.locator('[data-dashboard-section="kpi-grid"]').wait_for(state="visible", timeout=90_000)
    page.get_by_text("Surface Readiness").wait_for(timeout=90_000)
    page.locator(".dashboard-ready-marker").wait_for(state="attached", timeout=90_000)
    page.locator(".loading-panel").wait_for(state="detached", timeout=90_000)
    page.wait_for_load_state("networkidle", timeout=90_000)


def capture_screenshots(url: str, output_dir: Path, viewports: Iterable[str]) -> list[Path]:
    """Capture dashboard screenshots for the requested viewport names."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("SKIP dashboard visual regression: playwright is not installed")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            for name in viewports:
                width, height = VIEWPORTS[name]
                page.set_viewport_size({"width": width, "height": height})
                page.goto(url, wait_until="networkidle", timeout=90_000)
                _wait_for_dashboard_settled(page)
                path = output_dir / f"dashboard_{name}.png"
                page.screenshot(path=str(path), full_page=True, animations="disabled")
                paths.append(path)
        finally:
            browser.close()
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture Streamlit dashboard screenshots.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--output-dir", default="artifacts/dashboard_screenshots")
    parser.add_argument("--viewports", nargs="*", choices=sorted(VIEWPORTS), default=list(VIEWPORTS))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if not _has_playwright():
        delegated = _delegate_to_repo_venv(root)
        if delegated is not None:
            return delegated
    port = _pick_port(args.port)
    url = f"http://127.0.0.1:{port}"
    proc = _start_streamlit(port, root)
    try:
        _wait_for_http(url)
        paths = capture_screenshots(url, root / args.output_dir, args.viewports)
        if paths:
            print("Captured dashboard screenshots:")
            for path in paths:
                print(f" - {path}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
