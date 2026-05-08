"""Named deterministic demo option-chain provider."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Tuple

import pandas as pd

from src.data.options_provider import OptionsChainMetadata
from src.data.price_provider import RealTimePriceProvider
from src.data.synthetic_options import SyntheticOptionsGenerator


class DemoOptionsProvider:
    """Return deterministic synthetic option chains for demo/fallback mode."""

    source = "demo synthetic provider"
    mode = "Synthetic"

    def __init__(
        self,
        price_provider: RealTimePriceProvider,
        *,
        rate_provider: Any | None = None,
        dividend_provider: Any | None = None,
        random_seed: int = 1729,
        max_expirations: int = 8,
    ):
        self.random_seed = int(random_seed)
        self.max_expirations = max(1, int(max_expirations))
        self.generator = SyntheticOptionsGenerator(
            price_provider,
            rate_provider=rate_provider,
            dividend_provider=dividend_provider,
            demo_seed=self.random_seed,
        )

    def fetch_chain(
        self,
        symbol: str,
        spot_price: float,
        *,
        fallback_reason: str | None = None,
        as_of: datetime | None = None,
    ) -> Tuple[pd.DataFrame, OptionsChainMetadata]:
        """Return a deterministic demo chain and provenance metadata."""
        key = symbol.upper()
        timestamp = as_of or datetime.now()
        chain = self.generator.create_chain(key, spot_price=spot_price, as_of=timestamp)
        expirations = sorted(pd.to_datetime(chain["expiration"]).dropna().unique())
        selected = set(expirations[: self.max_expirations])
        chain = chain[pd.to_datetime(chain["expiration"]).isin(selected)].reset_index(drop=True)
        metadata = OptionsChainMetadata(
            symbol=key,
            source=self.source,
            mode=self.mode,
            timestamp=timestamp,
            expirations_requested=min(len(expirations), self.max_expirations),
            expirations_loaded=len(selected),
            raw_rows=len(chain),
            valid_rows=len(chain),
            rejected_rows=0,
            data_quality_score=100.0 if not chain.empty else 0.0,
            fallback_reason=fallback_reason,
            warnings=["Demo provider generated deterministic synthetic option data."],
        )
        return chain, metadata

    def representative_greeks(
        self,
        symbol: str,
        spot_price: float,
        *,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
    ) -> dict[str, float]:
        """Return deterministic representative Greeks for demo/fallback summaries."""
        return self.generator.calculate_greeks(
            symbol,
            spot_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )

    def cache_status(self) -> dict[str, Any]:
        return {"entries": 0, "source": self.source, "seed": self.random_seed}

    def clear_cache(self) -> None:
        return None
