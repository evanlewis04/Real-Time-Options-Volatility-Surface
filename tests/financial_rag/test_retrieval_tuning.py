"""Deterministic ranking tests for non-NVDA retrieval tuning.

These guard the capital-return / buyback ranking boost added so that
issuer-purchase evidence outranks generic "capital"/"return" filing chunks for
queries phrased as "capital return or buybacks" (common for banks and energy
issuers), without artificially boosting those chunks for unrelated queries.
"""

from src.financial_rag.retrieval.local_dense import lexical_relevance_score


REPURCHASE_TEXT = (
    "The board authorized common share repurchase activity, including issuer "
    "purchases of equity securities under the buyback program."
)
DIVIDEND_TEXT = "The board declared a quarterly dividend payable to shareholders."
GENERIC_CAPITAL_TEXT = (
    "Return on tangible common equity and the capital ratios improved across "
    "the reportable business segments during the year."
)

BUYBACK_QUERY = "What does the company say about capital return or buybacks?"
DEMAND_QUERY = "What does the company say about data center demand?"


def test_buyback_query_ranks_repurchase_text_over_generic_capital_text() -> None:
    repurchase = lexical_relevance_score(BUYBACK_QUERY, REPURCHASE_TEXT)
    generic = lexical_relevance_score(BUYBACK_QUERY, GENERIC_CAPITAL_TEXT)

    assert repurchase > generic


def test_buyback_query_prefers_repurchase_over_dividend_only_text() -> None:
    repurchase = lexical_relevance_score(BUYBACK_QUERY, REPURCHASE_TEXT)
    dividend = lexical_relevance_score(BUYBACK_QUERY, DIVIDEND_TEXT)

    # Tiered bonus: repurchase-specific language outranks dividend-only language.
    assert repurchase > dividend


def test_capital_return_boost_is_query_conditional() -> None:
    boosted = lexical_relevance_score(BUYBACK_QUERY, REPURCHASE_TEXT)
    unboosted = lexical_relevance_score(DEMAND_QUERY, REPURCHASE_TEXT)

    # The repurchase chunk only gets the capital-return boost when the query
    # actually asks about capital return / buybacks.
    assert boosted > unboosted
