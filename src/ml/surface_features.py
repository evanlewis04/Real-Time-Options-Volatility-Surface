"""Deterministic feature builders for research-only surface denoising.

The feature matrix describes observed quote rows and optional historical-prior
estimates. It does not relabel any row as a market observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.quant.surface_prior import HistoricalSurfacePrior


FEATURE_PROVENANCE = "ml_feature_set_research_inputs_not_market_observations"
PRIOR_FEATURE_PROVENANCE = "historical_prior_estimate_not_market_observation"

EXPIRY_BUCKETS = ("front", "medium", "deferred", "long")
OPTION_TYPES = ("call", "put")
PRICE_SOURCE_BUCKETS = ("midpoint", "mark", "last", "model", "other")


@dataclass(frozen=True)
class SurfaceFeatureSchema:
    """Stable ML feature schema used for training, prediction, and persistence."""

    version: str = "surface_ml_features_v1"
    feature_names: tuple[str, ...] = (
        "log_moneyness",
        "moneyness",
        "dte",
        "expiry_bucket_front",
        "expiry_bucket_medium",
        "expiry_bucket_deferred",
        "expiry_bucket_long",
        "option_type_call",
        "option_type_put",
        "bid_ask_spread_pct",
        "quote_age_days",
        "volume_log1p",
        "open_interest_log1p",
        "price_source_midpoint",
        "price_source_mark",
        "price_source_last",
        "price_source_model",
        "price_source_other",
        "forward_moneyness",
        "risk_free_rate",
        "dividend_yield",
        "event_flag",
        "historical_iv_prior",
        "raw_iv",
    )
    categorical_maps: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "expiry_bucket": EXPIRY_BUCKETS,
            "option_type": OPTION_TYPES,
            "selected_price_source": PRICE_SOURCE_BUCKETS,
        }
    )

    def as_dict(self) -> dict[str, Any]:
        """Return persistence-safe schema metadata."""
        return {
            "version": self.version,
            "feature_names": list(self.feature_names),
            "categorical_maps": {key: list(values) for key, values in self.categorical_maps.items()},
        }


@dataclass(frozen=True)
class SurfaceFeatureFrame:
    """Feature matrix plus supervised-learning target and provenance metadata."""

    features: pd.DataFrame
    target: pd.Series
    weights: pd.Series
    rows: pd.DataFrame
    schema: SurfaceFeatureSchema
    metadata: dict[str, Any]

    def empty(self) -> bool:
        return self.features.empty or self.target.empty


def build_surface_ml_features(
    chain: pd.DataFrame,
    spot: float,
    *,
    prior: HistoricalSurfacePrior | None = None,
    schema: SurfaceFeatureSchema | None = None,
    target_column: str = "computedIV",
    fit_eligible_only: bool = True,
) -> SurfaceFeatureFrame:
    """Return a deterministic numeric feature set for local denoising experiments."""
    active_schema = schema or SurfaceFeatureSchema()
    if chain.empty or spot <= 0.0:
        return _empty_feature_frame(active_schema, "No option rows or invalid spot")
    work = _prepared_rows(chain, spot, target_column, fit_eligible_only=fit_eligible_only)
    if work.empty:
        return _empty_feature_frame(active_schema, "No valid IV target rows")

    work["historical_iv_prior"] = _historical_prior_feature(work, prior)
    feature_rows = pd.DataFrame(index=work.index)
    feature_rows["log_moneyness"] = work["log_moneyness"]
    feature_rows["moneyness"] = work["moneyness"]
    feature_rows["dte"] = work["dte"]
    for bucket in EXPIRY_BUCKETS:
        feature_rows[f"expiry_bucket_{bucket}"] = (work["expiry_bucket"] == bucket).astype(float)
    for option_type in OPTION_TYPES:
        feature_rows[f"option_type_{option_type}"] = (work["option_type"] == option_type).astype(float)
    feature_rows["bid_ask_spread_pct"] = work["bid_ask_spread_pct"]
    feature_rows["quote_age_days"] = work["quote_age_days"]
    feature_rows["volume_log1p"] = np.log1p(work["volume"].clip(lower=0.0))
    feature_rows["open_interest_log1p"] = np.log1p(work["open_interest"].clip(lower=0.0))
    for source in PRICE_SOURCE_BUCKETS:
        feature_rows[f"price_source_{source}"] = (work["selected_price_source"] == source).astype(float)
    feature_rows["forward_moneyness"] = work["forward_moneyness"]
    feature_rows["risk_free_rate"] = work["risk_free_rate"]
    feature_rows["dividend_yield"] = work["dividend_yield"]
    feature_rows["event_flag"] = work["event_flag"]
    feature_rows["historical_iv_prior"] = work["historical_iv_prior"]
    feature_rows["raw_iv"] = work["raw_iv"]

    features = feature_rows.reindex(columns=active_schema.feature_names).astype(float)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
    target = work["target_iv"].astype(float).rename("target_iv")
    weights = work["fit_weight"].astype(float).rename("fit_weight")
    metadata = {
        "schema_version": active_schema.version,
        "feature_count": int(len(active_schema.feature_names)),
        "row_count": int(len(features)),
        "target_column": target_column,
        "fit_eligible_only": bool(fit_eligible_only),
        "provenance": FEATURE_PROVENANCE,
        "target_provenance": "observed_or_computed_current_chain_iv",
        "historical_prior_feature": bool(prior and prior.available),
        "historical_prior_provenance": PRIOR_FEATURE_PROVENANCE if prior else None,
        "estimate_warning": "ML denoising features are research inputs and are not market observations.",
    }
    return SurfaceFeatureFrame(
        features=features.reset_index(drop=True),
        target=target.reset_index(drop=True),
        weights=weights.reset_index(drop=True),
        rows=work.reset_index(drop=True),
        schema=active_schema,
        metadata=metadata,
    )


def _prepared_rows(
    chain: pd.DataFrame,
    spot: float,
    target_column: str,
    *,
    fit_eligible_only: bool,
) -> pd.DataFrame:
    target = target_column if target_column in chain else "impliedVolatility"
    required = {"strike", "daysToExpiration", target}
    if not required.issubset(chain.columns):
        return pd.DataFrame()
    out = pd.DataFrame(index=chain.index)
    out["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
    out["dte"] = pd.to_numeric(chain["daysToExpiration"], errors="coerce")
    out["target_iv"] = pd.to_numeric(chain[target], errors="coerce")
    out["raw_iv"] = _numeric_column(chain, "impliedVolatility", default=np.nan)
    out["log_moneyness"] = _log_moneyness(chain, spot)
    out["moneyness"] = np.exp(out["log_moneyness"])
    out["forward_moneyness"] = _forward_moneyness(chain, out["strike"], spot)
    out["bid_ask_spread_pct"] = _spread_pct(chain)
    out["quote_age_days"] = _numeric_column(chain, "quoteAgeSeconds", default=0.0) / 86400.0
    out["volume"] = _numeric_column(chain, "volume", default=0.0)
    out["open_interest"] = _numeric_column(chain, "openInterest", default=0.0)
    out["risk_free_rate"] = _numeric_column(chain, "riskFreeRate", default=0.0)
    out["dividend_yield"] = _dividend_yield(chain)
    out["event_flag"] = _event_flag(chain)
    out["fit_weight"] = _numeric_column(chain, "fitWeight", default=1.0).clip(lower=0.0)
    out["option_type"] = (
        chain.get("type", pd.Series("", index=chain.index)).astype(str).str.lower().where(lambda s: s.isin(OPTION_TYPES), "put")
    )
    source = chain.get("selectedPriceSource", chain.get("markSource", pd.Series("other", index=chain.index)))
    out["selected_price_source"] = source.astype(str).str.lower().map(_price_source_bucket)
    out["expiry_bucket"] = out["dte"].map(_expiry_bucket)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["strike", "dte", "target_iv", "log_moneyness"])
    out = out[(out["strike"] > 0.0) & (out["dte"] > 0.0) & (out["target_iv"] > 0.0)]
    if fit_eligible_only and "fitEligible" in chain:
        eligible = chain.loc[out.index, "fitEligible"].astype(bool)
        out = out.loc[eligible]
    return out.sort_values(["dte", "strike", "option_type"]).reset_index(drop=True)


def _historical_prior_feature(work: pd.DataFrame, prior: HistoricalSurfacePrior | None) -> pd.Series:
    fallback = pd.Series(np.nan, index=work.index, dtype=float)
    if prior is None or not prior.available or prior.grid.empty:
        return fallback
    grid = prior.grid.dropna(subset=["dte", "log_moneyness", "prior_iv"])
    if grid.empty:
        return fallback
    dte = grid["dte"].to_numpy(dtype=float)
    log_money = grid["log_moneyness"].to_numpy(dtype=float)
    prior_iv = grid["prior_iv"].to_numpy(dtype=float)
    dte_scale = max(float(np.nanmax(dte) - np.nanmin(dte)), 1.0)
    log_scale = max(float(np.nanmax(log_money) - np.nanmin(log_money)), 0.05)
    current = work[["dte", "log_moneyness"]].to_numpy(dtype=float)
    distances = ((current[:, None, 0] - dte[None, :]) / dte_scale) ** 2 + (
        (current[:, None, 1] - log_money[None, :]) / log_scale
    ) ** 2
    nearest = np.argmin(distances, axis=1)
    inside = (
        (current[:, 0] >= np.nanmin(dte))
        & (current[:, 0] <= np.nanmax(dte))
        & (current[:, 1] >= np.nanmin(log_money))
        & (current[:, 1] <= np.nanmax(log_money))
    )
    values = np.where(inside, prior_iv[nearest], np.nan)
    return pd.Series(values, index=work.index, dtype=float)


def _numeric_column(chain: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column not in chain:
        return pd.Series(default, index=chain.index, dtype=float)
    return pd.to_numeric(chain[column], errors="coerce").fillna(default).astype(float)


def _log_moneyness(chain: pd.DataFrame, spot: float) -> pd.Series:
    if "logMoneyness" in chain:
        values = pd.to_numeric(chain["logMoneyness"], errors="coerce")
        if values.notna().any():
            return values
    strikes = pd.to_numeric(chain["strike"], errors="coerce")
    if "forwardPrice" in chain:
        forwards = pd.to_numeric(chain["forwardPrice"], errors="coerce")
        return np.log(strikes / forwards.where(forwards > 0.0))
    return np.log(strikes / float(spot))


def _forward_moneyness(chain: pd.DataFrame, strikes: pd.Series, spot: float) -> pd.Series:
    if "forwardMoneyness" in chain:
        values = pd.to_numeric(chain["forwardMoneyness"], errors="coerce")
        if values.notna().any():
            return values.fillna(strikes / float(spot))
    if "forwardPrice" in chain:
        forwards = pd.to_numeric(chain["forwardPrice"], errors="coerce")
        return strikes / forwards.where(forwards > 0.0)
    return strikes / float(spot)


def _spread_pct(chain: pd.DataFrame) -> pd.Series:
    if "bidAskSpreadPct" in chain:
        return pd.to_numeric(chain["bidAskSpreadPct"], errors="coerce").fillna(0.0)
    if {"bid", "ask"}.issubset(chain.columns):
        bid = pd.to_numeric(chain["bid"], errors="coerce")
        ask = pd.to_numeric(chain["ask"], errors="coerce")
        mid = (bid + ask) / 2.0
        return ((ask - bid) / mid.where(mid > 0.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(0.0, index=chain.index, dtype=float)


def _dividend_yield(chain: pd.DataFrame) -> pd.Series:
    for column in ("effectiveDividendYield", "dividendYield"):
        if column in chain:
            values = pd.to_numeric(chain[column], errors="coerce")
            if values.notna().any():
                return values.fillna(0.0)
    return pd.Series(0.0, index=chain.index, dtype=float)


def _event_flag(chain: pd.DataFrame) -> pd.Series:
    for column in ("eventFlag", "hasEvent", "hasEarningsEvent", "event_count", "eventCount"):
        if column in chain:
            values = chain[column]
            if values.dtype == bool:
                return values.astype(float)
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.notna().any():
                return (numeric.fillna(0.0) > 0.0).astype(float)
    return pd.Series(0.0, index=chain.index, dtype=float)


def _expiry_bucket(dte: float) -> str:
    value = float(dte)
    if value <= 30.0:
        return "front"
    if value <= 90.0:
        return "medium"
    if value <= 180.0:
        return "deferred"
    return "long"


def _price_source_bucket(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"mid", "midpoint"}:
        return "midpoint"
    if text in {"mark", "last", "model"}:
        return text
    return "other"


def _empty_feature_frame(schema: SurfaceFeatureSchema, reason: str) -> SurfaceFeatureFrame:
    return SurfaceFeatureFrame(
        features=pd.DataFrame(columns=schema.feature_names),
        target=pd.Series(dtype=float, name="target_iv"),
        weights=pd.Series(dtype=float, name="fit_weight"),
        rows=pd.DataFrame(),
        schema=schema,
        metadata={
            "schema_version": schema.version,
            "feature_count": int(len(schema.feature_names)),
            "row_count": 0,
            "reason": reason,
            "provenance": FEATURE_PROVENANCE,
            "estimate_warning": "ML denoising features are research inputs and are not market observations.",
        },
    )
