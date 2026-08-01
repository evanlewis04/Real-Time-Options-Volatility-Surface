"""Eval-critical SEC filings pinned by accession number.

The default ingest path selects filings by *latest-N* per ticker, which drifts:
an exhibit or filing that backed a gold eval case can age out of the fetch
window, and a ticker can even resolve to a different registrant CIK than the one
that files the document the eval depends on. Because the local corpus is
gitignored, a fresh clone rebuilds it by fetching from SEC EDGAR, so without a
fixed anchor the eval numbers are not reproducible.

This module is the single source of truth for the handful of filings the gold
eval depends on. The ingest/repair path fetches these *additively*, on top of
the latest-N selection, so the eval-critical documents are always present
regardless of drift. Each pin is fetched under the CIK *derived from its own
accession number* (the first ten digits are the filer CIK by construction),
not the ticker->CIK lookup, so a decoy or successor-issuer ticker mapping cannot
divert the fetch away from the intended registrant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PinnedFiling:
    """A single eval-critical filing, anchored by accession number.

    ``ticker`` is the label the fetched documents are tagged under in the corpus
    (so gold labels for that ticker resolve). ``accession_number`` is the SEC
    accession; the filer CIK is derived from it, not looked up from the ticker.
    """

    ticker: str
    accession_number: str
    note: str


# Keep this minimal: exactly the filings the four coverage-gap gold cases need,
# nothing more. A general per-ticker pinning framework is intentionally out of
# scope.
PINNED_FILINGS: tuple[PinnedFiling, ...] = (
    PinnedFiling(
        ticker="NVDA",
        accession_number="0001045810-26-000051",
        note=(
            "NVDA earnings 8-K carrying the EX-99 press release (q1fy27pr.htm) and "
            "CFO commentary (q1fy27cfocommentary.htm). Backs gold cases "
            "nvda-press-release-gross-margin and nvda-cfo-revenue, which age out of "
            "the latest-N 8-K window."
        ),
    ),
    PinnedFiling(
        ticker="XOM",
        accession_number="0000034088-26-000045",
        note=(
            "Exxon Mobil Corp 10-K (xom-20251231.htm), filed under CIK 34088. The "
            "XOM ticker maps to CIK 2115436 (ExxonMobil Holdings Corp, the new "
            "public parent from the 2026 holding-company reorganization), which "
            "files no 10-K, so latest-N by ticker never reaches this filing. Backs "
            "gold cases xom-item1a-commodity-risk and xom-energy-transition."
        ),
    ),
)


def cik_from_accession(accession_number: str) -> int:
    """Return the filer CIK encoded in the first ten digits of an accession.

    SEC accession numbers are ``CCCCCCCCCC-YY-NNNNNN`` where the leading ten
    digits are the filer's zero-padded CIK by construction, so deriving the CIK
    this way is exact, not heuristic.
    """

    prefix = accession_number.split("-", 1)[0]
    if not prefix.isdigit():
        raise ValueError(f"Accession number {accession_number!r} has no numeric CIK prefix.")
    return int(prefix)


def pins_for_tickers(tickers: object) -> tuple[PinnedFiling, ...]:
    """Return the pins whose ticker is in ``tickers`` (case-insensitive)."""

    wanted = {str(ticker).strip().upper() for ticker in tickers}
    return tuple(pin for pin in PINNED_FILINGS if pin.ticker in wanted)
