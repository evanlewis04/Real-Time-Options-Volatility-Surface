"""Run the local engineering-quality verification workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CHECK_PATHS = [
    "src",
    "tests",
    "scripts",
]


def _run(label: str, args: list[str], cwd: Path) -> None:
    print(f"\n== {label} ==")
    subprocess.run(args, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run format, lint, compile, test, and brief smoke checks.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply Ruff formatting and safe lint fixes before verification.",
    )
    parser.add_argument(
        "--skip-healthcheck",
        action="store_true",
        help="Skip the RAG brief healthcheck for quicker inner-loop runs.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python = sys.executable

    if args.fix:
        _run("Format", [python, "-m", "ruff", "format", *CHECK_PATHS], root)
        _run("Lint fixes", [python, "-m", "ruff", "check", "--fix", *CHECK_PATHS], root)

    _run("Lint", [python, "-m", "ruff", "check", *CHECK_PATHS], root)
    _run(
        "Compile",
        [python, "-m", "compileall", "src", "tests", "scripts"],
        root,
    )
    _run("Tests", [python, "-m", "pytest", "-q"], root)
    if not args.skip_healthcheck:
        _run("Brief smoke", [python, "-m", "scripts.healthcheck"], root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
