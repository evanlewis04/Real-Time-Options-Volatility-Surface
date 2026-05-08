from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import FitFilterSettings
from src.quant.quote_quality import apply_quote_reliability_scores, score_quote
from tests.fixtures.noisy_option_chain import checked_clean_chain


def _atm_row() -> pd.Series:
    chain, _ = checked_clean_chain()
    mask = (
        (pd.to_datetime(chain["expiration"]).dt.strftime("%Y-%m-%d") == "2026-06-07")
        & (chain["strike"] == 200.0)
        & (chain["type"] == "call")
    )
    return chain[mask].iloc[0].copy()


def test_quote_reliability_scores_clean_atm_above_noisy_variants():
    reliable = _atm_row()
    reliable_score = score_quote(reliable, max_quote_age_days=5)

    stale = reliable.copy()
    stale["quoteAgeSeconds"] = 6 * 24 * 60 * 60
    stale["isStaleQuote"] = True

    last_only = reliable.copy()
    last_only["quoteQuality"] = "last_only"
    last_only["selectedPriceSource"] = "last"
    last_only["bid"] = 0.0
    last_only["ask"] = 0.0
    last_only["bidAskSpreadPct"] = np.nan

    wide = reliable.copy()
    wide["bidAskSpreadPct"] = 0.80
    wide["bidAskSpread"] = 1.60

    extreme = reliable.copy()
    extreme["moneyness"] = 2.30
    extreme["logMoneyness"] = float(np.log(2.30))

    no_arb = reliable.copy()
    no_arb["noArbitrageViolation"] = True
    no_arb["noArbitrageReasons"] = "calendar_monotonicity"

    rows = {
        "stale": score_quote(stale, max_quote_age_days=5),
        "last_only": score_quote(last_only, max_quote_age_days=5),
        "wide": score_quote(wide, max_quote_age_days=5),
        "extreme": score_quote(extreme, max_quote_age_days=5),
        "no_arb": score_quote(no_arb, max_quote_age_days=5),
    }

    assert reliable_score.score == 1.0
    assert all(result.score < reliable_score.score for result in rows.values())
    assert "stale_quote_penalty" in rows["stale"].penalty_reasons
    assert "last_only_penalty" in rows["last_only"].penalty_reasons
    assert "wide_spread_penalty" in rows["wide"].penalty_reasons
    assert "extreme_moneyness_penalty" in rows["extreme"].penalty_reasons
    assert "no_arbitrage_violation" in rows["no_arb"].hard_rejection_reasons
    assert rows["no_arb"].fit_eligible is False
    assert rows["no_arb"].display_eligible is True


def test_fit_filters_exclude_rows_without_hiding_display_eligibility():
    wide = _atm_row()
    wide["bidAskSpreadPct"] = 0.60
    wide["volume"] = 2
    wide["openInterest"] = 10

    result = score_quote(
        wide,
        fit_filters=FitFilterSettings(
            preset="Strict",
            max_bid_ask_spread_pct=0.35,
            min_volume=10,
            min_open_interest=50,
        ),
    )

    assert result.display_eligible is True
    assert result.fit_eligible is False
    assert "spread_above_fit_limit" in result.hard_rejection_reasons
    assert "volume_below_fit_minimum" in result.hard_rejection_reasons
    assert "open_interest_below_fit_minimum" in result.hard_rejection_reasons


def test_quote_reliability_annotations_include_reason_buckets_and_expiry_summary():
    chain, no_arb_meta = checked_clean_chain()
    work = chain[chain["strike"] == 200.0].head(4).copy()
    work.loc[work.index[0], "noArbitrageViolation"] = True
    work.loc[work.index[1], "quoteQuality"] = "last_only"
    work.loc[work.index[1], "selectedPriceSource"] = "last"
    work.loc[work.index[2], "bidAskSpreadPct"] = 0.75

    annotated, meta = apply_quote_reliability_scores(
        work,
        {
            "max_quote_age_days": 5,
            "expiry_quality": {"2026-05-22": {"valid_quotes": 4, "reason_buckets": {}}},
            **no_arb_meta,
        },
    )

    assert {"displayEligible", "quoteReliabilityScore", "fitWeight", "fitPenaltyReasons", "fitEligible"}.issubset(
        annotated.columns
    )
    assert int(annotated["displayEligible"].sum()) == 4
    assert int(annotated["fitEligible"].sum()) == 3
    assert meta["display_eligible_count"] == 4
    assert meta["fit_hard_rejection_reason_buckets"] == {"no_arbitrage_violation": 1}
    assert meta["fit_penalty_reason_buckets"]["last_only_penalty"] == 1
    assert meta["fit_penalty_reason_buckets"]["wide_spread_penalty"] == 1
    assert sum(item["fit_eligible_count"] for item in meta["expiry_reliability"].values()) == 3
    assert all("quote_reliability" in item for item in meta["expiry_quality"].values())


def test_diagnostic_raw_fit_policy_labels_no_arb_without_excluding():
    row = _atm_row()
    row["noArbitrageViolation"] = True

    result = score_quote(
        row,
        fit_filters=FitFilterSettings(preset="Diagnostic Raw", no_arbitrage_policy="allow"),
    )

    assert "no_arbitrage_penalty" in result.penalty_reasons
    assert "no_arbitrage_violation" not in result.hard_rejection_reasons
