"""Deterministic filing change detection over local chunks."""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass, asdict

from src.financial_rag.retrieval import LocalChunkRecord


@dataclass(frozen=True)
class FilingChangeRecord:
    change_id: str
    ticker: str
    item_number: str
    change_type: str
    previous_form_type: str
    current_form_type: str
    previous_filing_date: str
    current_filing_date: str
    previous_accession: str
    current_accession: str
    previous_chunk_id: str
    current_chunk_id: str
    previous_source_url: str
    current_source_url: str
    before_text: str
    after_text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def detect_filing_changes(
    chunks: list[LocalChunkRecord],
    *,
    ticker: str,
    forms: tuple[str, ...] = ("10-K", "10-Q"),
    max_records: int = 25,
) -> list[FilingChangeRecord]:
    """Compare sequential same-item filing chunks from local cache."""

    candidates = [
        chunk
        for chunk in chunks
        if _upper(chunk.metadata.get("ticker")) == ticker.upper()
        and _upper(chunk.metadata.get("form_type")) in forms
        and str(chunk.metadata.get("item_number", "")).strip()
    ]
    grouped: dict[tuple[str, str], list[LocalChunkRecord]] = {}
    for chunk in candidates:
        key = (str(chunk.metadata.get("form_type", "")), str(chunk.metadata.get("item_number", "")))
        grouped.setdefault(key, []).append(chunk)

    changes: list[FilingChangeRecord] = []
    for (_, item_number), item_chunks in sorted(grouped.items()):
        item_chunks.sort(key=lambda chunk: (str(chunk.metadata.get("filing_date", "")), chunk.chunk_id))
        for previous, current in zip(item_chunks, item_chunks[1:], strict=False):
            changes.extend(_chunk_changes(previous, current, item_number=item_number, ticker=ticker))
            if len(changes) >= max_records:
                return changes[:max_records]
    return changes[:max_records]


def _chunk_changes(
    previous: LocalChunkRecord,
    current: LocalChunkRecord,
    *,
    item_number: str,
    ticker: str,
) -> list[FilingChangeRecord]:
    before_parts = _paragraphs(previous.chunk_text)
    after_parts = _paragraphs(current.chunk_text)
    matcher = difflib.SequenceMatcher(a=before_parts, b=after_parts, autojunk=False)
    records: list[FilingChangeRecord] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before_text = "\n\n".join(before_parts[i1:i2])
        after_text = "\n\n".join(after_parts[j1:j2])
        records.append(
            _record(
                previous,
                current,
                ticker=ticker,
                item_number=item_number,
                change_type=tag,
                before_text=before_text,
                after_text=after_text,
            )
        )
    return records


def _record(
    previous: LocalChunkRecord,
    current: LocalChunkRecord,
    *,
    ticker: str,
    item_number: str,
    change_type: str,
    before_text: str,
    after_text: str,
) -> FilingChangeRecord:
    raw_id = "|".join(
        (
            ticker.upper(),
            item_number,
            previous.chunk_id,
            current.chunk_id,
            change_type,
            _compact(before_text)[:200],
            _compact(after_text)[:200],
        )
    )
    return FilingChangeRecord(
        change_id=hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24],
        ticker=ticker.upper(),
        item_number=item_number,
        change_type=change_type,
        previous_form_type=str(previous.metadata.get("form_type", "")),
        current_form_type=str(current.metadata.get("form_type", "")),
        previous_filing_date=str(previous.metadata.get("filing_date", "")),
        current_filing_date=str(current.metadata.get("filing_date", "")),
        previous_accession=str(previous.metadata.get("accession_number", "")),
        current_accession=str(current.metadata.get("accession_number", "")),
        previous_chunk_id=previous.chunk_id,
        current_chunk_id=current.chunk_id,
        previous_source_url=str(previous.metadata.get("source_url", "")),
        current_source_url=str(current.metadata.get("source_url", "")),
        before_text=before_text,
        after_text=after_text,
    )


def _paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if paragraphs:
        return paragraphs
    compact = _compact(text)
    return [compact] if compact else []


def _compact(text: str) -> str:
    return " ".join(text.split())


def _upper(value: object) -> str:
    return str(value or "").strip().upper()
