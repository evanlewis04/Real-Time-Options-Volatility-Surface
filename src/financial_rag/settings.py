"""Small configuration helpers for the Phase 1 filings smoke pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "your_",
    "example.com",
    "replace_me",
    "changeme",
    "placeholder",
)


def load_environment() -> None:
    """Load `.env` when present without overriding real environment values."""

    load_dotenv(override=False)


def configured_secret(name: str) -> str | None:
    """Return an environment value only when it is not an obvious placeholder."""

    value = os.getenv(name)
    if not value:
        return None
    stripped = value.strip()
    lowered = stripped.lower()
    if not stripped or any(marker in lowered for marker in PLACEHOLDER_MARKERS):
        return None
    return stripped


def project_root() -> Path:
    """Return the repository root from this source file."""

    return Path(__file__).resolve().parents[2]
