"""Citation and opt-in answer synthesis utilities."""

from .citations import CitationValidation, HydratedCitation, extract_citation_labels, validate_citations
from .openai_responses import (
    DEFAULT_OPENAI_SYNTHESIS_MODEL,
    EvidenceAnswer,
    OpenAIReadiness,
    check_openai_readiness,
    synthesize_answer_from_query_payload,
    validate_answer_citations,
)

__all__ = [
    "DEFAULT_OPENAI_SYNTHESIS_MODEL",
    "CitationValidation",
    "EvidenceAnswer",
    "HydratedCitation",
    "OpenAIReadiness",
    "check_openai_readiness",
    "extract_citation_labels",
    "synthesize_answer_from_query_payload",
    "validate_answer_citations",
    "validate_citations",
]
