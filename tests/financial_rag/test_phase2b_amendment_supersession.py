"""Phase 1 Stage 2b: section-level amendment supersession inside the as-of view.

The headline guarantee: within an as-of view at D, a section restated by an
amendment knowable as of D returns only the amendment's version — never the
pre-amendment original — while unamended sections of the original still return.
Supersession is section-level, so dropping the whole original is a bug this
pins against. Lineage is derived (EDGAR has no explicit amends-link), so the
fixtures below carry clean ``(cik, base_form, period_end)`` groups with known
section overlap, the deterministic vehicle for the invariant.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from src.financial_rag.amendment_supersession import superseded_chunk_ids
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever

D0 = "2026-01-10T12:00:00+00:00"  # original 10-K
D1 = "2026-03-15T12:00:00+00:00"  # 10-K/A restating Item 7, adding Item 15


class _ConstantEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _chunk(
    chunk_id: str,
    *,
    filed_at: str,
    form_type: str = "10-K",
    item_number: str | None = "7",
    section_path: str | None = None,
    cik: str = "0000320193",
    period_end: str = "2025-12-31",
    accession: str | None = None,
) -> LocalChunkRecord:
    metadata = {
        "chunk_id": chunk_id,
        "document_id": f"doc-{chunk_id}",
        "ticker": "AAPL",
        "cik": cik,
        "form_type": form_type,
        "period_end": period_end,
        "filed_at": filed_at,
        "filing_date": (filed_at[:10] or "2026-01-01"),
        "accession_number": accession or f"acc-{chunk_id}",
        "source_url": f"https://www.sec.gov/Archives/{chunk_id}.htm",
    }
    if item_number is not None:
        metadata["item_number"] = item_number
    if section_path is not None:
        metadata["section_path"] = section_path
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text="risk factors and results of operations commentary",
        metadata=metadata,
    )


def _amendment_corpus() -> tuple[list[LocalChunkRecord], dict[str, list[float]]]:
    """One lineage group: 10-K (Items 1A/7/8) then a 10-K/A restating Item 7 and adding Item 15."""

    chunks = [
        _chunk("o_1a", filed_at=D0, item_number="1A", accession="acc-orig"),
        _chunk("o_7", filed_at=D0, item_number="7", accession="acc-orig"),
        _chunk("o_8", filed_at=D0, item_number="8", accession="acc-orig"),
        _chunk("a_7", filed_at=D1, form_type="10-K/A", item_number="7", accession="acc-amend"),
        _chunk("a_15", filed_at=D1, form_type="10-K/A", item_number="15", accession="acc-amend"),
    ]
    embeddings = {chunk.chunk_id: [1.0, 0.0] for chunk in chunks}
    return chunks, embeddings


def _retriever(chunks: list[LocalChunkRecord], embeddings: dict[str, list[float]]) -> LocalDenseRetriever:
    return LocalDenseRetriever(chunks=chunks, embeddings=embeddings, query_embedder=_ConstantEmbedder())


def _served(retriever: LocalDenseRetriever, **kwargs) -> set[str]:
    results = retriever.search(query="risk factors results", top_k=99, **kwargs)
    return {result.chunk_id for result in results}


# --- Headline invariant -----------------------------------------------------


def test_amended_section_returns_only_the_amendment_unamended_original_survives() -> None:
    chunks, embeddings = _amendment_corpus()
    retriever = _retriever(chunks, embeddings)
    as_of = datetime.fromisoformat(D1) + timedelta(days=1)

    served = _served(retriever, as_of=as_of)

    # Amended Item 7: only the /A version, never the pre-amendment original.
    assert "a_7" in served
    assert "o_7" not in served
    # Unamended Items 1A/8 from the original still served — no document-level drop.
    assert {"o_1a", "o_8"}.issubset(served)
    # Item 15 added by the amendment is served.
    assert "a_15" in served


def test_amendment_not_yet_knowable_leaves_original_section_intact() -> None:
    # As of a date before the /A was filed, the original Item 7 is still served
    # and the amendment is invisible — supersession never reaches into the future.
    chunks, embeddings = _amendment_corpus()
    retriever = _retriever(chunks, embeddings)
    as_of = datetime.fromisoformat(D0) + timedelta(days=5)

    served = _served(retriever, as_of=as_of)

    assert {"o_1a", "o_7", "o_8"}.issubset(served)
    assert "a_7" not in served
    assert "a_15" not in served


def test_supersession_property_over_random_as_of_dates() -> None:
    # Over random as_of dates spanning before/between/after the two filings:
    # o_7 is served iff the amendment is NOT yet knowable; unamended o_1a/o_8 are
    # served whenever the original itself is knowable.
    chunks, embeddings = _amendment_corpus()
    retriever = _retriever(chunks, embeddings)
    d0 = datetime.fromisoformat(D0)
    d1 = datetime.fromisoformat(D1)
    span = (d1 - d0).total_seconds()

    rng = random.Random(20260801)
    for _ in range(200):
        as_of = d0 + timedelta(seconds=rng.uniform(-span * 0.5, span * 1.5))
        served = _served(retriever, as_of=as_of)
        amendment_knowable = d1 <= as_of
        original_knowable = d0 <= as_of

        if amendment_knowable:
            assert "o_7" not in served, f"served superseded o_7 at as_of={as_of}"
            assert "a_7" in served
        elif original_knowable:
            assert "o_7" in served, f"dropped live o_7 at as_of={as_of}"
            assert "a_7" not in served

        if original_knowable:
            assert {"o_1a", "o_8"}.issubset(served), f"dropped unamended section at as_of={as_of}"


# --- No-op / isolation ------------------------------------------------------


def test_as_of_none_is_a_pure_no_op_no_supersession() -> None:
    # Without as_of there is no as-of view, so nothing is superseded and the
    # original Item 7 is returned alongside the amendment — this is what keeps
    # the §4 eval reproducing exactly.
    chunks, embeddings = _amendment_corpus()
    retriever = _retriever(chunks, embeddings)

    served = _served(retriever)

    assert {"o_1a", "o_7", "o_8", "a_7", "a_15"} == served


def test_supersession_does_not_cross_lineage_groups() -> None:
    # A later filing under a different cik / period_end must not supersede.
    chunks, embeddings = _amendment_corpus()
    other = _chunk(
        "x_7",
        filed_at=D1,
        form_type="10-K/A",
        item_number="7",
        cik="0000789019",
        period_end="2024-06-30",
        accession="acc-other",
    )
    chunks.append(other)
    embeddings[other.chunk_id] = [1.0, 0.0]
    retriever = _retriever(chunks, embeddings)
    as_of = datetime.fromisoformat(D1) + timedelta(days=1)

    served = _served(retriever, as_of=as_of)

    # o_7 is superseded within its own group; x_7 (different group) is untouched.
    assert "o_7" not in served
    assert "x_7" in served


# --- Honest fallback (missing / unreliable section metadata) ----------------


def test_amendment_with_missing_section_metadata_does_not_supersede() -> None:
    # A /A chunk with no item_number and no section_path: we cannot tell what it
    # restates, so the original section is retained (both kept), never silently
    # dropped.
    original = _chunk("o_only", filed_at=D0, item_number="1A", accession="acc-orig")
    amend_blank = _chunk(
        "a_blank",
        filed_at=D1,
        form_type="10-K/A",
        item_number=None,
        section_path=None,
        accession="acc-amend",
    )
    chunks = [original, amend_blank]
    superseded = superseded_chunk_ids(chunks)

    assert superseded == set()


def test_tie_on_filed_at_retains_both_versions() -> None:
    # Two versions of the same section at the identical filed_at: ambiguous
    # ordering, so neither is dropped.
    a = _chunk("t_a", filed_at=D1, item_number="7", accession="acc-a")
    b = _chunk("t_b", filed_at=D1, form_type="10-K/A", item_number="7", accession="acc-b")
    superseded = superseded_chunk_ids([a, b])

    assert superseded == set()


def test_missing_lineage_component_never_participates() -> None:
    # No period_end -> lineage key unresolvable -> chunk cannot be superseded and
    # cannot supersede, even against a same-cik later filing.
    original = _chunk("n_orig", filed_at=D0, item_number="7", period_end="", accession="acc-orig")
    amend = _chunk("n_amend", filed_at=D1, form_type="10-K/A", item_number="7", period_end="", accession="acc-amend")
    superseded = superseded_chunk_ids([original, amend])

    assert superseded == set()


def test_section_path_used_when_item_number_absent() -> None:
    # When item_number is missing, section_path is the section identity.
    original = _chunk(
        "p_orig",
        filed_at=D0,
        item_number=None,
        section_path="Item 7. MD&A",
        accession="acc-orig",
    )
    amend = _chunk(
        "p_amend",
        filed_at=D1,
        form_type="10-K/A",
        item_number=None,
        section_path="Item 7. MD&A",
        accession="acc-amend",
    )
    superseded = superseded_chunk_ids([original, amend])

    assert superseded == {"p_orig"}
