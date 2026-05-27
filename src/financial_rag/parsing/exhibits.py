"""EX-99 exhibit classification helpers."""

from __future__ import annotations

import re


EXHIBIT_TYPE_PRESS_RELEASE = "PRESS_RELEASE"
EXHIBIT_TYPE_CFO_COMMENTARY = "CFO_COMMENTARY"
EXHIBIT_TYPE_PREPARED_REMARKS = "PREPARED_REMARKS"
EXHIBIT_TYPE_PRESENTATION = "PRESENTATION"
EXHIBIT_TYPE_GENERIC = "EX-99"


def classify_ex99_exhibit(
    *,
    filename: str = "",
    description: str = "",
    text: str = "",
) -> str:
    """Classify an EX-99 document using filename, SEC description, and text hints."""

    combined = " ".join(part for part in (filename, description, _text_sample(text)) if part).lower()
    compact = re.sub(r"[^a-z0-9]+", " ", combined)

    if any(marker in compact for marker in ("cfo commentary", "chief financial officer commentary")):
        return EXHIBIT_TYPE_CFO_COMMENTARY
    if "cfo" in compact and "commentary" in compact:
        return EXHIBIT_TYPE_CFO_COMMENTARY
    if any(marker in compact for marker in ("prepared remarks", "prepared remark", "remarks prepared")):
        return EXHIBIT_TYPE_PREPARED_REMARKS
    if any(marker in compact for marker in ("slide", "slides", "presentation", "presentat")):
        return EXHIBIT_TYPE_PRESENTATION
    if _looks_like_press_release(compact):
        return EXHIBIT_TYPE_PRESS_RELEASE
    return EXHIBIT_TYPE_GENERIC


def _looks_like_press_release(value: str) -> bool:
    return any(
        marker in value
        for marker in (
            "press release",
            "pressreleasedated",
            "news release",
            "earnings release",
            "financial results",
            "quarterly results",
        )
    ) or re.search(r"(^| )(pr|pressreleasedated)( |$)", value) is not None


def _text_sample(text: str) -> str:
    if not text:
        return ""
    return text[:4000]
