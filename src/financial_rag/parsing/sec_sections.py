"""SEC-aware section parsing for filings and exhibit text."""

from __future__ import annotations

import re
from dataclasses import dataclass


SECTION_PARSER_VERSION = "sec_sections_v3"

# Minimum fraction of the document the detected item sections must cover. Some
# filings (e.g. Intel's 10-K) only carry literal "Item N." labels in a trailing
# cross-reference index while the body uses bare headings ("Risk Factors").
# Building sections from those labels would drop the entire body before the
# first match, so when coverage falls below this floor we fall back to
# whole-document chunking instead of discarding the body.
MIN_SECTION_COVERAGE = 0.5

_TENK_Q_ITEM_RE = re.compile(
    r"(?im)(?:^|(?<=\n))[ \t]*(?:part\s+[ivx]+\s+)?item\s+"
    r"(?P<number>(?:1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16))"
    r"\s*[\.\-:]*\s+(?P<title>[^\n]{3,160})$"
)
_EIGHTK_ITEM_RE = re.compile(
    r"(?im)(?:^|(?<=\n))[ \t]*item\s+(?P<number>\d\.\d{2})\s*[\.\-:]*\s+(?P<title>[^\n]{3,180})$"
)


@dataclass(frozen=True)
class ParsedSection:
    """A filing section with source offsets in the extracted text."""

    item_number: str
    title: str
    section_path: str
    text: str
    start_offset: int
    end_offset: int


def parse_sec_sections(text: str, *, form_type: str) -> list[ParsedSection]:
    """Return SEC item sections when item boundaries are detectable.

    The parser is intentionally small and conservative. It recognizes common
    10-K/10-Q item headings and 8-K item headings, then falls back to a single
    whole-document section when boundaries are absent.
    """

    text = _normalize_item_heading_lines(text)
    normalized_form = form_type.upper()
    if normalized_form in {"10-K", "10-Q"}:
        matches = _dedupe_heading_matches(_probable_heading_matches(list(_TENK_Q_ITEM_RE.finditer(text))))
    elif normalized_form == "8-K":
        matches = _dedupe_heading_matches(_probable_heading_matches(list(_EIGHTK_ITEM_RE.finditer(text))))
    else:
        matches = []

    if not matches:
        return [_fallback_section(text)]

    sections: list[ParsedSection] = []
    for index, match in enumerate(matches):
        end_offset = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start() : end_offset].strip()
        if not body:
            continue
        number = _normalize_item_number(match.group("number"))
        title = _clean_heading(match.group("title"))
        sections.append(
            ParsedSection(
                item_number=number,
                title=title,
                section_path=f"Item {number}. {title}",
                text=body,
                start_offset=match.start(),
                end_offset=end_offset,
            )
        )
    if not sections:
        return [_fallback_section(text)]
    if _section_coverage(sections, len(text)) < MIN_SECTION_COVERAGE:
        # The detected headings only cover a small tail of the document (a
        # trailing cross-reference index or stray labels); keep the body intact.
        return [_fallback_section(text)]
    return sections


def _section_coverage(sections: list[ParsedSection], total_length: int) -> float:
    if total_length <= 0:
        return 1.0
    captured = sum(len(section.text) for section in sections)
    return captured / total_length


def _dedupe_heading_matches(matches: list[re.Match[str]]) -> list[re.Match[str]]:
    """Drop repeated table-of-contents-style headings before real sections."""

    if len(matches) < 2:
        return matches
    deduped: list[re.Match[str]] = []
    seen: set[tuple[str, str]] = set()
    for match in reversed(matches):
        key = (_normalize_item_number(match.group("number")), _clean_heading(match.group("title")).lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return list(reversed(deduped))


def _probable_heading_matches(matches: list[re.Match[str]]) -> list[re.Match[str]]:
    return [match for match in matches if _is_probable_heading(match.group("title"))]


def _fallback_section(text: str) -> ParsedSection:
    return ParsedSection(
        item_number="",
        title="",
        section_path="",
        text=text.strip(),
        start_offset=0,
        end_offset=len(text),
    )


def _normalize_item_number(value: str) -> str:
    return value.strip().upper()


def _clean_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .:-\t")


def _is_probable_heading(title: str) -> bool:
    cleaned = _clean_heading(title)
    lower = cleaned.lower()
    if any(
        marker in lower
        for marker in (
            "for a discussion",
            "for additional information",
            "in this annual report",
            "in this quarterly report",
            "for further discussion",
        )
    ):
        return False
    if cleaned.startswith(("”", '"', "'", "“")) or cleaned.endswith(("”", "“")):
        return False
    return True


def _normalize_item_heading_lines(text: str) -> str:
    """Put common SEC item headings on their own line before regex parsing."""

    return re.sub(
        r"(?i)(?<!\n)(\bitem\s+(?:1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16|\d\.\d{2})\s*[\.\-:]+\s+)",
        r"\n\1",
        text,
    )
