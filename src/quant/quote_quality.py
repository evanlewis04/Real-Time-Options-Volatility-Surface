"""Row-level quote reliability scoring for surface fitting.

The scorer annotates observed quotes with deterministic fit weights and reason
labels. It does not alter market prices or replace observed volatility inputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


RELIABILITY_COLUMNS = [
    "quoteReliabilityScore",
    "fitWeight",
    "fitPenaltyReasons",
    "fitHardRejectionReasons",
    "fitEligible",
]

LOW_SCORE_THRESHOLD = 0.50
MIN_FIT_SCORE = 0.25


@dataclass(frozen=True)
class QuoteReliabilityScore:
    """Scoring result for one observed option quote."""

    score: float
    fit_weight: float
    penalty_reasons: tuple[str, ...]
    hard_rejection_reasons: tuple[str, ...]
    fit_eligible: bool


def apply_quote_reliability_scores(
    chain: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Annotate option rows with reliability scores and expiry summaries."""
    meta = metadata or {}
    if chain.empty:
        return _empty_frame(chain), _empty_metadata(meta)

    max_quote_age_days = _positive_float(meta.get("max_quote_age_days"), default=5.0)
    rows: list[QuoteReliabilityScore] = [
        score_quote(row, max_quote_age_days=max_quote_age_days) for _, row in chain.iterrows()
    ]

    out = chain.copy()
    out["quoteReliabilityScore"] = [row.score for row in rows]
    out["fitWeight"] = [row.fit_weight for row in rows]
    out["fitPenaltyReasons"] = [";".join(row.penalty_reasons) for row in rows]
    out["fitHardRejectionReasons"] = [";".join(row.hard_rejection_reasons) for row in rows]
    out["fitEligible"] = [row.fit_eligible for row in rows]

    summary = quote_reliability_summary(out, meta)
    return out, summary


def score_quote(row: pd.Series, *, max_quote_age_days: float = 5.0) -> QuoteReliabilityScore:
    """Score one quote from 0.0 to 1.0 using observed row-level quality signals."""
    score = 1.0
    penalties: list[str] = []
    hard_rejections: list[str] = []

    penalty, reason = _spread_penalty(row)
    score -= penalty
    if reason:
        penalties.append(reason)

    penalty, reason = _quote_age_penalty(row, max_quote_age_days=max_quote_age_days)
    score -= penalty
    if reason:
        penalties.append(reason)

    penalty, reason = _last_only_penalty(row)
    score -= penalty
    if reason:
        penalties.append(reason)

    penalty, reason = _liquidity_penalty(row)
    score -= penalty
    if reason:
        penalties.append(reason)

    penalty, reason = _moneyness_penalty(row)
    score -= penalty
    if reason:
        penalties.append(reason)

    penalty, reason, hard = _iv_penalty(row)
    score -= penalty
    if reason:
        penalties.append(reason)
    if hard:
        hard_rejections.append(hard)

    penalty, reason, hard = _selected_price_penalty(row)
    score -= penalty
    if reason:
        penalties.append(reason)
    if hard:
        hard_rejections.append(hard)

    penalty, reason, hard = _expiry_penalty(row)
    score -= penalty
    if reason:
        penalties.append(reason)
    if hard:
        hard_rejections.append(hard)

    if _bool(row.get("noArbitrageViolation")):
        score -= 0.55
        penalties.append("no_arbitrage_penalty")
        hard_rejections.append("no_arbitrage_violation")

    if _bool(row.get("parityViolation")):
        score -= 0.25
        penalties.append("parity_penalty")

    computed_iv = _float(row.get("computedIV"))
    if "computedIV" in row.index and not _valid_iv(computed_iv):
        score -= 0.30
        penalties.append("computed_iv_missing_penalty")
        hard_rejections.append("computed_iv_missing")

    score = round(float(np.clip(score, 0.0, 1.0)), 4)
    penalty_reasons = tuple(dict.fromkeys(penalties))
    hard_rejection_reasons = tuple(dict.fromkeys(hard_rejections))
    fit_eligible = score >= MIN_FIT_SCORE and not hard_rejection_reasons
    return QuoteReliabilityScore(
        score=score,
        fit_weight=score if fit_eligible else 0.0,
        penalty_reasons=penalty_reasons,
        hard_rejection_reasons=hard_rejection_reasons,
        fit_eligible=fit_eligible,
    )


def quote_reliability_summary(chain: pd.DataFrame, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build chain-level and expiry-level reliability metadata."""
    existing_expiry_quality = dict((metadata or {}).get("expiry_quality") or {})
    if chain.empty or "quoteReliabilityScore" not in chain:
        return _empty_metadata({"expiry_quality": existing_expiry_quality})

    scores = pd.to_numeric(chain["quoteReliabilityScore"], errors="coerce")
    fit_eligible = _bool_series(chain.get("fitEligible"), index=chain.index)
    penalty_buckets = _reason_buckets(chain.get("fitPenaltyReasons"))
    hard_buckets = _reason_buckets(chain.get("fitHardRejectionReasons"))
    summary = {
        "quote_reliability_summary": {
            "median_score": _rounded_or_none(scores.median()),
            "mean_score": _rounded_or_none(scores.mean()),
            "low_score_count": int((scores < LOW_SCORE_THRESHOLD).sum()),
            "fit_eligible_count": int(fit_eligible.sum()),
            "fit_excluded_count": int((~fit_eligible).sum()),
            "penalty_reason_buckets": penalty_buckets,
            "hard_rejection_reason_buckets": hard_buckets,
        },
        "fit_penalty_reason_buckets": penalty_buckets,
        "fit_hard_rejection_reason_buckets": hard_buckets,
        "expiry_reliability": _expiry_reliability(chain),
    }
    summary["expiry_quality"] = _expiry_quality_with_reliability(
        existing_expiry_quality,
        summary["expiry_reliability"],
    )
    return summary


def _spread_penalty(row: pd.Series) -> tuple[float, str | None]:
    spread_pct = _float(row.get("bidAskSpreadPct"))
    if spread_pct is None:
        return 0.0, None
    if spread_pct <= 0.05:
        return 0.0, None
    penalty = min(0.35, 0.35 * (spread_pct - 0.05) / 0.45)
    return float(penalty), "wide_spread_penalty"


def _quote_age_penalty(row: pd.Series, *, max_quote_age_days: float) -> tuple[float, str | None]:
    age_seconds = _float(row.get("quoteAgeSeconds"))
    if age_seconds is None or age_seconds <= 0:
        return 0.0, None
    max_age_seconds = max(max_quote_age_days, 1.0 / 24.0) * 24 * 60 * 60
    ratio = age_seconds / max_age_seconds
    if ratio <= 0.10:
        return 0.0, None
    penalty = min(0.25, 0.25 * (ratio - 0.10) / 0.90)
    return float(penalty), "stale_quote_penalty"


def _last_only_penalty(row: pd.Series) -> tuple[float, str | None]:
    quality = str(row.get("quoteQuality") or "").lower()
    source = str(row.get("selectedPriceSource") or row.get("markSource") or "").lower()
    if "last_only" in quality:
        return 0.25, "last_only_penalty"
    if source == "last":
        return 0.10, "selected_last_price_penalty"
    return 0.0, None


def _liquidity_penalty(row: pd.Series) -> tuple[float, str | None]:
    volume = _float(row.get("volume"))
    open_interest = _float(row.get("openInterest"))
    penalty = 0.0
    if volume is None or volume < 1:
        penalty += 0.08
    elif volume < 10:
        penalty += 0.04
    if open_interest is None or open_interest < 1:
        penalty += 0.08
    elif open_interest < 50:
        penalty += 0.04
    return penalty, "low_liquidity_penalty" if penalty else None


def _moneyness_penalty(row: pd.Series) -> tuple[float, str | None]:
    distance = _log_moneyness_distance(row)
    if distance is None or distance <= 0.35:
        return 0.0, None
    penalty = min(0.25, 0.25 * (distance - 0.35) / 0.25)
    return float(penalty), "extreme_moneyness_penalty"


def _iv_penalty(row: pd.Series) -> tuple[float, str | None, str | None]:
    raw_iv = _float(row.get("impliedVolatility"))
    if not _valid_iv(raw_iv):
        return 0.30, "provider_iv_invalid_penalty", "provider_iv_invalid"
    if raw_iv is not None and (raw_iv > 1.50 or raw_iv < 0.03):
        return 0.20, "provider_iv_suspicious_penalty", None
    return 0.0, None, None


def _selected_price_penalty(row: pd.Series) -> tuple[float, str | None, str | None]:
    price = _float(row.get("selectedMarketPrice"))
    if "selectedMarketPrice" in row.index and (price is None or price <= 0):
        return 0.40, "selected_price_unavailable_penalty", "selected_price_unavailable"
    source = str(row.get("selectedPriceSource") or "").lower()
    if source == "unavailable":
        return 0.25, "selected_price_unavailable_penalty", "selected_price_unavailable"
    return 0.0, None, None


def _expiry_penalty(row: pd.Series) -> tuple[float, str | None, str | None]:
    dte = _float(row.get("daysToExpiration"))
    if dte is not None and dte <= 0:
        return 1.0, "expired_contract_penalty", "expired_contract"
    if dte is not None and dte < 2:
        return 0.15, "near_expiry_penalty", None
    return 0.0, None, None


def _expiry_reliability(chain: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "expiration" not in chain:
        return {}
    out: dict[str, dict[str, Any]] = {}
    expiries = pd.to_datetime(chain["expiration"], errors="coerce").dt.normalize()
    for expiry, group in chain.groupby(expiries, dropna=True):
        scores = pd.to_numeric(group["quoteReliabilityScore"], errors="coerce")
        fit_eligible = _bool_series(group.get("fitEligible"), index=group.index)
        penalty_buckets = _reason_buckets(group.get("fitPenaltyReasons"))
        out[expiry.date().isoformat()] = {
            "median_score": _rounded_or_none(scores.median()),
            "low_score_count": int((scores < LOW_SCORE_THRESHOLD).sum()),
            "fit_eligible_count": int(fit_eligible.sum()),
            "excluded_count": int((~fit_eligible).sum()),
            "dominant_penalty_reasons": [
                {"reason": reason, "count": count}
                for reason, count in Counter(penalty_buckets).most_common(3)
            ],
            "penalty_reason_buckets": penalty_buckets,
        }
    return dict(sorted(out.items()))


def _expiry_quality_with_reliability(
    existing_expiry_quality: dict[str, Any],
    expiry_reliability: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    out = {str(expiry): dict(payload) for expiry, payload in existing_expiry_quality.items()}
    for expiry, payload in expiry_reliability.items():
        entry = out.setdefault(expiry, {})
        entry["quote_reliability"] = dict(payload)
    return dict(sorted(out.items()))


def _reason_buckets(values: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if values is None:
        return {}
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    for value in series.dropna():
        for reason in str(value).split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] += 1
    return dict(sorted(counts.items()))


def _empty_frame(chain: pd.DataFrame) -> pd.DataFrame:
    out = chain.copy()
    for column in RELIABILITY_COLUMNS:
        if column not in out:
            out[column] = pd.Series(dtype="bool" if column == "fitEligible" else "float64")
    return out


def _empty_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    expiry_quality = metadata.get("expiry_quality")
    return {
        "quote_reliability_summary": {
            "median_score": None,
            "mean_score": None,
            "low_score_count": 0,
            "fit_eligible_count": 0,
            "fit_excluded_count": 0,
            "penalty_reason_buckets": {},
            "hard_rejection_reason_buckets": {},
        },
        "fit_penalty_reason_buckets": {},
        "fit_hard_rejection_reason_buckets": {},
        "expiry_reliability": {},
        "expiry_quality": dict(expiry_quality or {}),
    }


def _log_moneyness_distance(row: pd.Series) -> float | None:
    log_moneyness = _float(row.get("logMoneyness"))
    if log_moneyness is not None:
        return abs(log_moneyness)
    moneyness = _float(row.get("moneyness"))
    if moneyness is not None and moneyness > 0:
        return abs(float(np.log(moneyness)))
    return None


def _valid_iv(value: float | None) -> bool:
    return value is not None and np.isfinite(value) and 0.01 < value < 5.0


def _bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value)


def _bool_series(values: Any, *, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(False, index=index)
    return pd.Series(values, index=index).fillna(False).astype(bool)


def _float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _positive_float(value: Any, *, default: float) -> float:
    number = _float(value)
    if number is None or number <= 0:
        return default
    return number


def _rounded_or_none(value: Any) -> float | None:
    number = _float(value)
    return round(number, 4) if number is not None else None
