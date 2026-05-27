"""SEC document text extraction and lightweight section parsing."""

from .exhibits import classify_ex99_exhibit
from .sec_sections import SECTION_PARSER_VERSION, ParsedSection, parse_sec_sections
from .sec_text import PARSER_VERSION, extract_readable_text

__all__ = [
    "PARSER_VERSION",
    "SECTION_PARSER_VERSION",
    "ParsedSection",
    "classify_ex99_exhibit",
    "extract_readable_text",
    "parse_sec_sections",
]
