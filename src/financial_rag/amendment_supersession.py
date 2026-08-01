"""Amendment supersession for point-in-time as-of views (Phase 1 Stage 2b).

Within an as-of view at instant D, an amendment (e.g. a ``10-K/A``) supersedes
only the *sections* it restates — never the whole original filing. Dropping the
entire original when a ``/A`` exists would silently lose unamended content (the
Items that live only in the original), so supersession here is **section-level**.

EDGAR publishes no machine-readable "amends accession Y" pointer, so lineage is
*derived*: filings are grouped by ``(cik, base_form, period_end)`` where
``base_form`` is ``form_type`` with a trailing ``/A`` stripped (``10-K/A`` →
``10-K``). Within a lineage group, for each section key (``item_number``, else
``section_path``) only the latest-``filed_at`` version's chunks survive; chunks
from strictly-earlier versions of that same section are superseded.

Honest fallback (documented, not silent): any chunk whose lineage key, section
key, or ``filed_at`` cannot be resolved does not participate — it is never
dropped and can never cause a drop. Where item/section metadata is missing or
unreliable we cannot tell what a ``/A`` supersedes, so we retain both rather
than silently drop valid content or silently serve a section we *know* was
superseded.

The caller passes only the as-of-*knowable* chunks (``filed_at <= as_of``), so a
version filed after D never supersedes anything visible at D.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable

from src.financial_rag.retrieval.local_dense import _normalize, _parse_utc_datetime

if TYPE_CHECKING:
    from src.financial_rag.retrieval.local_dense import LocalChunkRecord


def _base_form(form_type: Any) -> str:
    """``form_type`` with a single trailing ``/A`` amendment suffix removed."""

    normalized = _normalize(form_type)
    if normalized.endswith("/A"):
        return normalized[:-2].strip()
    return normalized


def _lineage_key(metadata: dict[str, Any]) -> tuple[str, str, str] | None:
    """Derived amendment-lineage key, or ``None`` when it cannot be resolved.

    All three components must be present: without ``cik``/``period_end`` we
    cannot prove two filings are the same filing across an amendment, and
    without a ``base_form`` there is nothing to group. A ``None`` here means the
    chunk sits outside any supersession decision (honest fallback: retained).
    """

    cik = _normalize(metadata.get("cik"))
    base_form = _base_form(metadata.get("form_type"))
    period_end = _normalize(metadata.get("period_end"))
    if not cik or not base_form or not period_end:
        return None
    return (cik, base_form, period_end)


def _section_key(metadata: dict[str, Any]) -> str | None:
    """Stable identity of the section a chunk belongs to, or ``None``.

    Prefers ``item_number`` (the amended unit EDGAR restates), falling back to
    ``section_path``. ``None`` when neither is present — the honest-fallback
    case where we cannot tell what a ``/A`` would supersede, so the chunk is
    retained.
    """

    item_number = _normalize(metadata.get("item_number"))
    if item_number:
        return f"item:{item_number}"
    section_path = _normalize(metadata.get("section_path"))
    if section_path:
        return f"path:{section_path}"
    return None


def superseded_chunk_ids(chunks: Iterable["LocalChunkRecord"]) -> set[str]:
    """Chunk ids superseded by a later-filed amendment of the same section.

    ``chunks`` must already be restricted to the as-of-knowable set
    (``filed_at <= as_of``). For each ``(lineage, section)`` the latest
    ``filed_at`` wins; strictly-earlier versions of that section are superseded.
    Ties on ``filed_at`` are retained (never dropped) so nothing valid is lost
    on ambiguous ordering.
    """

    # lineage key -> section key -> list of (chunk_id, filed_at)
    groups: dict[tuple[str, str, str], dict[str, list[tuple[str, datetime]]]] = {}
    for chunk in chunks:
        metadata = chunk.metadata
        lineage = _lineage_key(metadata)
        if lineage is None:
            continue
        section = _section_key(metadata)
        if section is None:
            continue
        filed_at = _parse_utc_datetime(metadata.get("filed_at"))
        if filed_at is None:
            continue
        groups.setdefault(lineage, {}).setdefault(section, []).append((chunk.chunk_id, filed_at))

    superseded: set[str] = set()
    for sections in groups.values():
        for entries in sections.values():
            latest = max(filed_at for _, filed_at in entries)
            for chunk_id, filed_at in entries:
                if filed_at < latest:
                    superseded.add(chunk_id)
    return superseded
