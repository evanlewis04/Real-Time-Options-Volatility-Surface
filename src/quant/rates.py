"""Risk-free rate curves for option pricing and IV calculations."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_CURVE_PATH = Path("config/risk_free_curve.csv")
DEFAULT_CURVE_POINTS: tuple[tuple[int, float, str], ...] = (
    (1, 0.0525, "overnight"),
    (7, 0.0520, "1w"),
    (30, 0.0510, "1m"),
    (90, 0.0495, "3m"),
    (180, 0.0475, "6m"),
    (365, 0.0450, "1y"),
    (730, 0.0425, "2y"),
    (1825, 0.0400, "5y"),
    (3650, 0.0410, "10y"),
)


@dataclass(frozen=True)
class RatePoint:
    """One annualized continuously-compounded zero-rate point."""

    tenor_days: int
    rate: float
    label: str = ""


@dataclass(frozen=True)
class RateLookup:
    """Rate selected for a specific option maturity."""

    dte: int
    rate: float
    source: str
    mode: str
    tenor_days: int


@dataclass(frozen=True)
class RateCurve:
    """Interpolated annualized risk-free zero-rate curve."""

    as_of: datetime
    source: str
    mode: str
    points: tuple[RatePoint, ...]
    fallback_reason: Optional[str] = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def rate_for_dte(self, dte: int | float | None) -> RateLookup:
        """Return the interpolated annualized rate for calendar days to expiry."""
        clean_dte = int(max(0, round(float(dte or 0))))
        if not self.points:
            point = RatePoint(30, 0.05, "fallback")
            return RateLookup(clean_dte, point.rate, self.source, self.mode, point.tenor_days)

        ordered = sorted(self.points, key=lambda point: point.tenor_days)
        tenors = np.array([point.tenor_days for point in ordered], dtype=float)
        rates = np.array([point.rate for point in ordered], dtype=float)
        rate = float(np.interp(clean_dte, tenors, rates, left=rates[0], right=rates[-1]))
        tenor_idx = int(np.argmin(np.abs(tenors - clean_dte)))
        return RateLookup(clean_dte, rate, self.source, self.mode, int(tenors[tenor_idx]))

    def discount_factor(self, dte: int | float | None) -> float:
        """Return the continuously-compounded discount factor for a maturity."""
        lookup = self.rate_for_dte(dte)
        return float(np.exp(-lookup.rate * lookup.dte / 365.0))

    def metadata_dict(self) -> dict[str, Any]:
        """Return dashboard-friendly provenance and curve details."""
        return {
            "risk_free_rate_source": self.source,
            "risk_free_rate_mode": self.mode,
            "risk_free_rate_timestamp": self.as_of,
            "risk_free_rate_fallback_reason": self.fallback_reason,
            "risk_free_rate_curve": [
                {"tenor_days": point.tenor_days, "rate": point.rate, "label": point.label}
                for point in self.points
            ],
            "risk_free_rate_warnings": list(self.warnings),
        }


class RateSourceError(RuntimeError):
    """Raised when a rate source cannot produce a usable curve."""


class LocalCurveRateSource:
    """Load a configurable risk-free curve from a local CSV file."""

    def __init__(self, path: Path | str = DEFAULT_CURVE_PATH):
        self.path = Path(path)

    def load_curve(self) -> RateCurve:
        """Load the local curve, falling back to built-in points if absent."""
        if not self.path.exists():
            return _default_curve(
                source=f"local:{self.path}",
                fallback_reason="Local risk-free curve file missing; using built-in fallback curve",
            )

        frame = pd.read_csv(self.path)
        points = tuple(_points_from_frame(frame))
        if not points:
            return _default_curve(
                source=f"local:{self.path}",
                fallback_reason="Local risk-free curve file had no usable rows; using built-in fallback curve",
            )

        return RateCurve(
            as_of=datetime.fromtimestamp(self.path.stat().st_mtime),
            source=f"local:{self.path}",
            mode="Local",
            points=points,
        )


class FredTreasuryRateSource:
    """Optional live Treasury curve loader using FRED's API."""

    SERIES = {
        30: "DGS1MO",
        90: "DGS3MO",
        180: "DGS6MO",
        365: "DGS1",
        730: "DGS2",
        1825: "DGS5",
        3650: "DGS10",
    }

    def __init__(self, api_key: str | None = None, timeout_seconds: float = 5.0):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.timeout_seconds = timeout_seconds

    def load_curve(self) -> RateCurve:
        """Fetch latest constant-maturity Treasury rates from FRED."""
        if not self.api_key:
            raise RateSourceError("FRED_API_KEY is not configured")

        points: list[RatePoint] = []
        warnings: list[str] = []
        for tenor_days, series_id in self.SERIES.items():
            try:
                value = self._fetch_latest_percent(series_id)
                points.append(RatePoint(tenor_days=tenor_days, rate=value / 100.0, label=series_id))
            except Exception as exc:
                warnings.append(f"{series_id}: {exc}")

        if len(points) < 2:
            raise RateSourceError("FRED did not return enough usable Treasury tenors")

        return RateCurve(
            as_of=datetime.now(),
            source="FRED Treasury constant maturity",
            mode="Live",
            points=tuple(sorted(points, key=lambda point: point.tenor_days)),
            warnings=tuple(warnings),
        )

    def _fetch_latest_percent(self, series_id: str) -> float:
        params = urllib.parse.urlencode(
            {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            }
        )
        url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
        with urllib.request.urlopen(url, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))

        for observation in payload.get("observations", []):
            value = observation.get("value")
            if value and value != ".":
                return float(value)
        raise RateSourceError(f"No numeric observation for {series_id}")


class RiskFreeRateProvider:
    """Cached risk-free curve provider with offline-safe local fallback."""

    def __init__(
        self,
        preferred_source: str | None = None,
        local_curve_path: Path | str = DEFAULT_CURVE_PATH,
        cache_ttl_seconds: int = 3600,
    ):
        self.preferred_source = (preferred_source or os.getenv("ROVS_RISK_FREE_RATE_SOURCE") or "local").lower()
        self.local_source = LocalCurveRateSource(local_curve_path)
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cached: tuple[RateCurve, datetime] | None = None

    def get_curve(self, force_refresh: bool = False) -> RateCurve:
        """Return a cached curve unless refresh is requested or TTL expired."""
        now = datetime.now()
        if self._cached and not force_refresh and now - self._cached[1] < self.cache_ttl:
            return self._cached[0]

        curve = self._load_curve()
        self._cached = (curve, now)
        return curve

    def clear_cache(self) -> None:
        """Clear the cached rate curve."""
        self._cached = None

    def _load_curve(self) -> RateCurve:
        if self.preferred_source in {"fred", "live", "treasury"}:
            try:
                return FredTreasuryRateSource().load_curve()
            except Exception as exc:
                local = self.local_source.load_curve()
                return RateCurve(
                    as_of=local.as_of,
                    source=local.source,
                    mode="Fallback",
                    points=local.points,
                    fallback_reason=f"Live FRED Treasury curve unavailable: {exc}",
                    warnings=local.warnings,
                )
        return self.local_source.load_curve()


def apply_curve_to_options(frame: pd.DataFrame, curve: RateCurve) -> pd.DataFrame:
    """Attach expiry-specific risk-free rates to a normalized option chain."""
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    dte = pd.to_numeric(enriched.get("daysToExpiration"), errors="coerce")
    enriched["riskFreeRate"] = dte.apply(lambda value: curve.rate_for_dte(value).rate if pd.notna(value) else np.nan)
    return enriched


def expiry_rate_metadata(frame: pd.DataFrame, curve: RateCurve) -> dict[str, float]:
    """Build an expiry-date to rate map from an options chain."""
    if frame.empty or "expiration" not in frame.columns:
        return {}

    expirations = pd.to_datetime(frame["expiration"], errors="coerce")
    dtes = pd.to_numeric(frame.get("daysToExpiration"), errors="coerce")
    out: dict[str, float] = {}
    for expiry, dte in zip(expirations, dtes):
        if pd.isna(expiry) or pd.isna(dte):
            continue
        out[expiry.date().isoformat()] = curve.rate_for_dte(float(dte)).rate
    return dict(sorted(out.items()))


def _points_from_frame(frame: pd.DataFrame) -> Iterable[RatePoint]:
    lower_cols = {str(col).strip().lower(): col for col in frame.columns}
    tenor_col = lower_cols.get("tenor_days") or lower_cols.get("days") or lower_cols.get("tenor")
    rate_col = lower_cols.get("rate") or lower_cols.get("annualized_rate") or lower_cols.get("zero_rate")
    label_col = lower_cols.get("label")
    if tenor_col is None or rate_col is None:
        return []

    points: list[RatePoint] = []
    for _, row in frame.iterrows():
        try:
            tenor_days = _parse_tenor_days(row[tenor_col])
            rate = _parse_rate(row[rate_col])
            if tenor_days > 0 and -0.05 <= rate <= 0.25:
                label = str(row[label_col]) if label_col and pd.notna(row[label_col]) else str(row[tenor_col])
                points.append(RatePoint(tenor_days=tenor_days, rate=rate, label=label))
        except (TypeError, ValueError):
            continue
    return sorted(points, key=lambda point: point.tenor_days)


def _parse_tenor_days(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("missing tenor")
    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip().lower()
    if text.endswith("mo"):
        return int(float(text[:-2]) * 30)
    if text.endswith("m"):
        return int(float(text[:-1]) * 30)
    if text.endswith("w"):
        return int(float(text[:-1]) * 7)
    if text.endswith("y"):
        return int(float(text[:-1]) * 365)
    if text.endswith("d"):
        return int(float(text[:-1]))
    return int(float(text))


def _parse_rate(value: Any) -> float:
    if pd.isna(value):
        raise ValueError("missing rate")
    text = str(value).strip().replace("%", "")
    rate = float(text)
    return rate / 100.0 if abs(rate) > 1.0 else rate


def _default_curve(source: str = "built-in fallback curve", fallback_reason: str | None = None) -> RateCurve:
    return RateCurve(
        as_of=datetime.now(),
        source=source,
        mode="Fallback",
        points=tuple(RatePoint(days, rate, label) for days, rate, label in DEFAULT_CURVE_POINTS),
        fallback_reason=fallback_reason,
    )

