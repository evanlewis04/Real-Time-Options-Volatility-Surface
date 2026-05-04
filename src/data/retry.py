"""Small retry helper for rate-limited or flaky market-data calls."""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar


T = TypeVar("T")
logger = logging.getLogger(__name__)


class DataProviderRetryError(RuntimeError):
    """Raised when a provider call exhausts retries."""


def call_with_backoff(
    fn: Callable[[], T],
    *,
    label: str,
    attempts: int = 3,
    base_delay: float = 0.25,
    max_delay: float = 2.0,
) -> T:
    """Call ``fn`` with bounded exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            logger.debug("%s failed on attempt %s/%s: %s", label, attempt, attempts, exc)
            time.sleep(delay)
    raise DataProviderRetryError(f"{label} failed after {attempts} attempts: {last_exc}") from last_exc
