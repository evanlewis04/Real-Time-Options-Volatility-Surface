"""SEC-aware chunking strategies for filings and EX-99 exhibits."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from src.financial_rag.models import DocumentChunk, FilingMetadata
from src.financial_rag.parsing import (
    PARSER_VERSION,
    SECTION_PARSER_VERSION,
    ParsedSection,
    parse_sec_sections,
)


CHUNKER_VERSION = "sec_aware_v2"
DEFAULT_MAX_CHARS = 2200
DEFAULT_OVERLAP_CHARS = 200

_SPEAKER_RE = re.compile(
    r"(?m)^[ \t]*(?P<speaker>[A-Z][A-Za-z .,'&-]{1,64})\s*[:\-]\s*(?P<rest>\S.*)?$"
)


@dataclass(frozen=True)
class TextBlock:
    text: str
    start_offset: int
    end_offset: int
    section_path: str = ""
    item_number: str = ""
    speaker_name: str = ""
    speaker_role: str = ""


def chunk_sec_document(
    text: str,
    metadata: FilingMetadata,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[DocumentChunk]:
    """Create stable chunks using filing-aware or exhibit-aware boundaries."""

    normalized = text.strip()
    if not normalized:
        return []

    if _is_exhibit(metadata):
        blocks = _exhibit_blocks(normalized)
    elif metadata.form_type.upper() in {"10-K", "10-Q", "8-K"}:
        blocks = _filing_item_blocks(normalized, metadata.form_type)
    else:
        blocks = [TextBlock(text=normalized, start_offset=0, end_offset=len(normalized))]

    chunks: list[DocumentChunk] = []
    for block in blocks:
        chunks.extend(
            _split_block(
                block,
                metadata,
                ordinal_start=len(chunks),
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )
    return chunks


def _is_exhibit(metadata: FilingMetadata) -> bool:
    return metadata.document_role == "exhibit" or metadata.form_type.upper() == "EX-99"


def _filing_item_blocks(text: str, form_type: str) -> list[TextBlock]:
    sections = parse_sec_sections(text, form_type=form_type)
    return [
        TextBlock(
            text=section.text,
            start_offset=section.start_offset,
            end_offset=section.end_offset,
            section_path=section.section_path,
            item_number=section.item_number,
        )
        for section in sections
        if section.text
    ]


def _exhibit_blocks(text: str) -> list[TextBlock]:
    speaker_blocks = _speaker_turn_blocks(text)
    if speaker_blocks:
        return speaker_blocks
    heading_blocks = _heading_paragraph_blocks(text)
    return heading_blocks or [TextBlock(text=text, start_offset=0, end_offset=len(text))]


def _speaker_turn_blocks(text: str) -> list[TextBlock]:
    matches = list(_SPEAKER_RE.finditer(text))
    obvious = [match for match in matches if _is_obvious_speaker(match.group("speaker"))]
    if len(obvious) < 2:
        return []

    blocks: list[TextBlock] = []
    for index, match in enumerate(obvious):
        start = match.start()
        body_start = match.end("speaker")
        end = obvious[index + 1].start() if index + 1 < len(obvious) else len(text)
        speaker = _clean_inline(match.group("speaker"))
        body = text[body_start:end].strip(" :-\n\t")
        if body:
            blocks.append(
                TextBlock(
                    text=f"{speaker}: {body}",
                    start_offset=start,
                    end_offset=end,
                    section_path=f"Speaker: {speaker}",
                    speaker_name=speaker,
                )
            )
    return blocks


def _heading_paragraph_blocks(text: str) -> list[TextBlock]:
    paragraphs = _paragraphs_with_offsets(text)
    blocks: list[TextBlock] = []
    current_heading = ""
    buffer: list[tuple[str, int, int]] = []

    for paragraph, start, end in paragraphs:
        if _looks_like_heading(paragraph):
            blocks.extend(_flush_paragraph_buffer(buffer, current_heading))
            buffer = []
            current_heading = _clean_inline(paragraph)
            continue
        buffer.append((paragraph, start, end))
    blocks.extend(_flush_paragraph_buffer(buffer, current_heading))
    return blocks


def _flush_paragraph_buffer(
    buffer: list[tuple[str, int, int]],
    heading: str,
) -> list[TextBlock]:
    if not buffer:
        return []
    text = "\n\n".join(paragraph for paragraph, _, _ in buffer)
    return [
        TextBlock(
            text=f"{heading}\n\n{text}" if heading else text,
            start_offset=buffer[0][1],
            end_offset=buffer[-1][2],
            section_path=heading,
        )
    ]


def _split_block(
    block: TextBlock,
    metadata: FilingMetadata,
    *,
    ordinal_start: int,
    max_chars: int,
    overlap_chars: int,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(block.text):
        target_end = min(start + max_chars, len(block.text))
        end = _choose_boundary(block.text, start=start, target_end=target_end)
        chunk_text = block.text[start:end].strip()
        if chunk_text:
            chunks.append(
                _build_chunk(
                    chunk_text=chunk_text,
                    metadata=metadata,
                    start=block.start_offset + start,
                    end=block.start_offset + end,
                    ordinal=ordinal_start + len(chunks),
                    section_path=block.section_path,
                    item_number=block.item_number,
                    speaker_name=block.speaker_name,
                    speaker_role=block.speaker_role,
                )
            )
        if end >= len(block.text):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def _build_chunk(
    *,
    chunk_text: str,
    metadata: FilingMetadata,
    start: int,
    end: int,
    ordinal: int,
    section_path: str,
    item_number: str,
    speaker_name: str,
    speaker_role: str,
) -> DocumentChunk:
    chunk_hash = hashlib.sha256(
        f"{metadata.document_id}|{start}|{end}|{section_path}|{speaker_name}|{chunk_text}".encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return DocumentChunk(
        chunk_id=f"{metadata.document_id}-chunk-{ordinal:04d}-{chunk_hash}",
        document_id=metadata.document_id,
        ticker=metadata.ticker,
        cik=metadata.cik,
        accession_number=metadata.accession_number,
        form_type=metadata.form_type,
        filing_date=metadata.filing_date,
        source_url=metadata.source_url,
        local_path=metadata.local_path,
        document_role=metadata.document_role,
        exhibit_type=metadata.exhibit_type,
        chunk_text=chunk_text,
        start_offset=start,
        end_offset=end,
        token_count=_approx_token_count(chunk_text),
        section_path=section_path,
        item_number=item_number,
        speaker_name=speaker_name,
        speaker_role=speaker_role,
        filed_at=metadata.filed_at,
        period_end=metadata.period_end,
        metadata={
            "parser_version": PARSER_VERSION,
            "section_parser_version": SECTION_PARSER_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "report_date": metadata.report_date,
            "document_name": metadata.document_name,
            "description": metadata.description,
            "content_hash": metadata.content_hash,
            "section_path": section_path,
            "item_number": item_number,
            "speaker_name": speaker_name,
            "speaker_role": speaker_role,
        },
    )


def _choose_boundary(text: str, *, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)
    window = text[start:target_end]
    paragraph_break = window.rfind("\n\n")
    if paragraph_break > max(400, int(len(window) * 0.45)):
        return start + paragraph_break
    sentence_break = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if sentence_break > max(400, int(len(window) * 0.55)):
        return start + sentence_break + 1
    return target_end


def _paragraphs_with_offsets(text: str) -> list[tuple[str, int, int]]:
    paragraphs: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, flags=re.S):
        paragraph = re.sub(r"[ \t]+", " ", match.group(0)).strip()
        if paragraph:
            paragraphs.append((paragraph, match.start(), match.end()))
    return paragraphs


def _looks_like_heading(paragraph: str) -> bool:
    cleaned = _clean_inline(paragraph)
    if not cleaned or len(cleaned) > 100:
        return False
    if cleaned.endswith((".", ";", ",")):
        return False
    words = cleaned.split()
    if len(words) > 10:
        return False
    upper_ratio = sum(1 for char in cleaned if char.isupper()) / max(
        1, sum(1 for char in cleaned if char.isalpha())
    )
    return upper_ratio > 0.65 or cleaned.istitle()


def _is_obvious_speaker(value: str) -> bool:
    cleaned = _clean_inline(value)
    if len(cleaned) > 64 or len(cleaned.split()) > 5:
        return False
    if cleaned.lower() in {"item", "table", "note", "page"}:
        return False
    return bool(re.search(r"[A-Za-z]", cleaned)) and (cleaned.istitle() or cleaned.isupper())


def _clean_inline(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" :-\t")


def _approx_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def sections_from_text(text: str, *, form_type: str) -> Iterable[ParsedSection]:
    """Small public helper for tests and notebooks."""

    return parse_sec_sections(text, form_type=form_type)
