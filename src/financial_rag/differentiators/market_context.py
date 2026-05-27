"""Optional market-context hook for filings reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class MarketContext:
    status: str
    ticker: str
    source_mode: str = ""
    message: str = ""
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = self.metrics or {}
        return payload


def get_market_context(
    ticker: str,
    *,
    provider: Callable[[str], dict[str, Any]] | None = None,
) -> MarketContext:
    """Attach optional volatility context without changing dashboard flow."""

    if provider is None:
        return MarketContext(
            status="unavailable",
            ticker=ticker.upper(),
            message="No market context provider was supplied.",
        )
    try:
        payload = provider(ticker.upper())
    except Exception as exc:  # pragma: no cover - defensive adapter boundary
        return MarketContext(
            status="unavailable",
            ticker=ticker.upper(),
            message=f"Market context provider failed: {exc}",
        )
    return MarketContext(
        status="ok",
        ticker=ticker.upper(),
        source_mode=str(payload.get("source_mode", payload.get("mode", ""))),
        message=str(payload.get("message", "Loaded market context.")),
        metrics=dict(payload.get("metrics", payload)),
    )
