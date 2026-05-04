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
                page.get_by_text("Options Volatility Surface Workstation").wait_for(timeout=90_000)
                path = output_dir / f"dashboard_{name}.png"
                page.screenshot(path=str(path), full_page=True)
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
