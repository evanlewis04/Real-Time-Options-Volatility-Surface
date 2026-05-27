"""Citation validation over retrieved local chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.financial_rag.retrieval import RetrievalResult


_CITATION_RE = re.compile(r"\[?(S\d+)\]?", flags=re.I)


@dataclass(frozen=True)
class HydratedCitation:
    label: str
    ticker: str
    form_type: str
    filing_date: str
    accession: str
    source_url: str
    chunk_id: str


@dataclass(frozen=True)
class CitationValidation:
    accepted: list[HydratedCitation]
    rejected: list[str]


def extract_citation_labels(answer_text: str) -> list[str]:
    """Extract unique S-style citation labels from text in encounter order."""

    labels: list[str] = []
    seen: set[str] = set()
    for match in _CITATION_RE.finditer(answer_text):
        label = match.group(1).upper()
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def validate_citations(
    candidate_labels: Iterable[str],
    retrieved_chunks: Iterable[RetrievalResult],
) -> CitationValidation:
    """Accept only labels that map to the supplied retrieval results."""

    by_label = {result.citation_label.upper(): result for result in retrieved_chunks}
    accepted: list[HydratedCitation] = []
    rejected: list[str] = []
    for raw_label in candidate_labels:
        label = _normalize_label(raw_label)
        result = by_label.get(label)
        if result is None:
            rejected.append(label)
            continue
        metadata = result.metadata
        accepted.append(
            HydratedCitation(
                label=label,
                ticker=str(metadata.get("ticker", "")),
                form_type=str(metadata.get("form_type", "")),
                filing_date=str(metadata.get("filing_date", "")),
                accession=str(metadata.get("accession_number", "")),
                source_url=result.source_url,
                chunk_id=result.chunk_id,
            )
        )
    return CitationValidation(accepted=accepted, rejected=rejected)


def _normalize_label(value: str) -> str:
    match = _CITATION_RE.search(value.strip())
    return match.group(1).upper() if match else value.strip().upper()
