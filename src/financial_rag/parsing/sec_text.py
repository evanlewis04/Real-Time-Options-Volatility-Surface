"""Simple readable-text extraction for SEC HTML and text documents."""

from __future__ import annotations

import re
from html import unescape

from bs4 import BeautifulSoup


PARSER_VERSION = "sec_text_v1"


def extract_readable_text(content: bytes | str) -> str:
    """Extract readable text without attempting SEC section parsing."""

    raw = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
    if _looks_like_html(raw):
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
    else:
        text = raw
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_html(value: str) -> bool:
    sample = value[:2048].lower()
    return "<html" in sample or "<body" in sample or "<table" in sample or "<div" in sample
