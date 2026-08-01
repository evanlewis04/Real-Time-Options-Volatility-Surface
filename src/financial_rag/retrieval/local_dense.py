"""Dense-only local retrieval over cached chunk and Voyage vector files."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Protocol

from src.financial_rag.storage import LocalRagStore

if TYPE_CHECKING:
    from src.financial_rag.corpus_snapshot import CorpusSnapshot


class QueryEmbeddingProvider(Protocol):
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        """Return dense query vectors."""


@dataclass(frozen=True)
class RetrievalFilters:
    ticker: str | None = None
    form_type: str | None = None
    form_types: tuple[str, ...] = ()
    accession: str | None = None
    document_role: str | None = None
    document_roles: tuple[str, ...] = ()
    exhibit_type: str | None = None
    exhibit_types: tuple[str, ...] = ()
    item_number: str | None = None
    item_numbers: tuple[str, ...] = ()
    speaker_name: str | None = None
    speaker_role: str | None = None


@dataclass(frozen=True)
class LocalChunkRecord:
    chunk_id: str
    chunk_text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    rank: int
    dense_score: float
    citation_label: str
    source_url: str
    source_excerpt: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class LocalDenseRetriever:
    """Cosine retriever over already cached chunk embeddings."""

    def __init__(
        self,
        *,
        chunks: list[LocalChunkRecord],
        embeddings: dict[str, list[float]],
        query_embedder: QueryEmbeddingProvider | None = None,
    ) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.query_embedder = query_embedder

    def search(
        self,
        *,
        query: str | None = None,
        query_vector: list[float] | None = None,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
        as_of: datetime | None = None,
    ) -> list[RetrievalResult]:
        if query_vector is None:
            if query is None:
                raise ValueError("Either query or query_vector is required.")
            if self.query_embedder is None:
                raise ValueError("A query embedder is required when query_vector is not provided.")
            query_vector = self.query_embedder.embed_queries([query])[0]

        # Point-in-time (Phase 1 Stage 2): when as_of is set, hard-filter the
        # candidate set to filings knowable as of that instant BEFORE scoring, so
        # look-ahead records never enter retrieval. as_of is None on the eval path,
        # making this a pure no-op there. Normalize as_of to UTC once, up front.
        as_of_utc = _coerce_utc(as_of) if as_of is not None else None

        # Amendment supersession (Phase 1 Stage 2b): inside the as-of view, an
        # amendment supersedes only the SECTIONS it restates. Compute the dropped
        # chunk ids over the as-of-knowable set (later-filed versions of a section
        # win) so a pre-amendment section is never served. Only runs when as_of is
        # set, keeping the as_of=None eval path a pure no-op.
        superseded: set[str] = set()
        if as_of_utc is not None:
            from src.financial_rag.amendment_supersession import superseded_chunk_ids

            knowable = [c for c in self.chunks if _is_knowable_as_of(c.metadata, as_of_utc)]
            superseded = superseded_chunk_ids(knowable)

        query_text = query or ""
        scored: list[tuple[float, LocalChunkRecord]] = []
        for chunk in self.chunks:
            if not _matches_filters(chunk.metadata, filters):
                continue
            if as_of_utc is not None and not _is_knowable_as_of(chunk.metadata, as_of_utc):
                continue
            if chunk.chunk_id in superseded:
                continue
            vector = self.embeddings.get(chunk.chunk_id)
            dense_score = cosine_similarity(query_vector, vector) if vector is not None else 0.0
            lexical_score = lexical_relevance_score(query_text, chunk.chunk_text, chunk.metadata)
            score = lexical_score + (dense_score * 0.25)
            if _is_safe_harbor_only(chunk.chunk_text) and _asks_for_operating_risks(query_text):
                score -= 1.25
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        results: list[RetrievalResult] = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            metadata = dict(chunk.metadata)
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    rank=rank,
                    dense_score=dense_score,
                    citation_label=f"S{rank}",
                    source_url=str(metadata.get("source_url", "")),
                    source_excerpt=_excerpt(chunk.chunk_text),
                    score=score,
                    metadata=metadata,
                )
            )
        return results


def load_local_retrieval_corpus(
    *,
    root: Path | str = Path("."),
    snapshot: "CorpusSnapshot | None" = None,
) -> tuple[list[LocalChunkRecord], dict[str, list[float]]]:
    """Load local chunk JSONL files and vector-cache JSON files.

    When ``snapshot`` is provided, the chunk set is restricted to the documents
    recorded in that snapshot at their recorded content (Phase 1 Stage 3). This
    is the pin seam: a pinned read reproduces even after new docs land.
    ``snapshot=None`` is a pure no-op, so the §4 eval reproduces exactly.
    """

    store = LocalRagStore(root=root)
    chunks = _load_chunks(store.chunks_dir)
    if snapshot is not None:
        from src.financial_rag.corpus_snapshot import restrict_chunks_to_snapshot

        chunks = restrict_chunks_to_snapshot(chunks, snapshot)
    embeddings = _load_embeddings(store.vector_cache_dir)
    return chunks, embeddings


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def lexical_relevance_score(query: str, text: str, metadata: dict[str, Any] | None = None) -> float:
    """Small sparse/BM25-style score used as a local fallback and rerank signal."""

    if not query or not text:
        return 0.0
    query_terms = _content_terms(query)
    text_lower = text.lower()
    text_terms = set(_content_terms(text_lower))
    if not query_terms or not text_terms:
        return 0.0

    overlap = sum(1 for term in query_terms if term in text_terms) / len(query_terms)
    phrase_bonus = sum(0.35 for phrase in _query_phrases(query) if phrase in text_lower)
    phrase_bonus += _domain_phrase_bonus(query, text_lower)
    metadata_bonus = _metadata_relevance_bonus(query, metadata or {})
    return overlap + phrase_bonus + metadata_bonus


def _load_chunks(chunks_dir: Path) -> list[LocalChunkRecord]:
    records: list[LocalChunkRecord] = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        for row in _read_jsonl(path):
            chunk_id = str(row.get("chunk_id", ""))
            if not chunk_id:
                continue
            records.append(
                LocalChunkRecord(
                    chunk_id=chunk_id,
                    chunk_text=str(row.get("chunk_text", "")),
                    metadata=_chunk_metadata(row),
                )
            )
    return records


def _load_embeddings(vector_cache_dir: Path) -> dict[str, list[float]]:
    embeddings: dict[str, list[float]] = {}
    for path in sorted(vector_cache_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunk_id = str(payload.get("chunk_id", ""))
        vector = payload.get("embedding", [])
        if chunk_id and isinstance(vector, list):
            embeddings[chunk_id] = [float(value) for value in vector]
    return embeddings


def _chunk_metadata(row: dict[str, Any]) -> dict[str, Any]:
    nested = row.get("metadata")
    metadata = dict(nested) if isinstance(nested, dict) else {}
    for key in (
        "chunk_id",
        "document_id",
        "ticker",
        "cik",
        "accession_number",
        "form_type",
        "filing_date",
        "source_url",
        "local_path",
        "document_role",
        "exhibit_type",
        "start_offset",
        "end_offset",
        "token_count",
        "section_path",
        "item_number",
        "speaker_name",
        "speaker_role",
        "filed_at",
        "period_end",
    ):
        if key in row and row[key] not in (None, ""):
            metadata[key] = row[key]
    return metadata


def _matches_filters(metadata: dict[str, Any], filters: RetrievalFilters | None) -> bool:
    if filters is None:
        return True
    checks = {
        "ticker": filters.ticker,
        "form_type": filters.form_type,
        "accession_number": filters.accession,
        "document_role": filters.document_role,
        "exhibit_type": filters.exhibit_type,
        "item_number": filters.item_number,
        "speaker_name": filters.speaker_name,
        "speaker_role": filters.speaker_role,
    }
    for key, expected in checks.items():
        if expected is None:
            continue
        actual = metadata.get(key, "")
        if key == "speaker_role" and not str(actual).strip() and _normalize(metadata.get("exhibit_type")) == _normalize(
            "CFO_COMMENTARY"
        ):
            continue
        if _normalize(actual) != _normalize(expected):
            return False
    list_checks = {
        "form_type": filters.form_types,
        "document_role": filters.document_roles,
        "exhibit_type": filters.exhibit_types,
        "item_number": filters.item_numbers,
    }
    for key, expected_values in list_checks.items():
        if not expected_values:
            continue
        actual = metadata.get(key, "")
        normalized_expected = {_normalize(value) for value in expected_values}
        if _normalize(actual) not in normalized_expected:
            return False
    return True


def _is_knowable_as_of(metadata: dict[str, Any], as_of_utc: datetime) -> bool:
    """True when the chunk's filing was public at or before ``as_of_utc``.

    A chunk with an empty or unparseable ``filed_at`` is *excluded* from an
    as-of view: a record whose public date cannot be proven is not knowable as
    of any date. It re-enters once the point-in-time backfill fills filed_at.
    """

    filed_at = _parse_utc_datetime(metadata.get("filed_at"))
    if filed_at is None:
        return False
    return filed_at <= as_of_utc


def _parse_utc_datetime(value: Any) -> datetime | None:
    """Parse a stored ISO-8601 ``filed_at`` into a tz-aware UTC datetime.

    Matches the format written by ``parse_acceptance_datetime`` (ISO-8601 with a
    ``+00:00`` offset, or a trailing ``Z``). Returns None for empty/unparseable
    input so the caller can apply the data-honesty exclusion policy.
    """

    text = str(value or "").strip()
    if not text:
        return None
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return _coerce_utc(parsed)


def _coerce_utc(value: datetime) -> datetime:
    """Normalize a datetime to UTC; a naive datetime is assumed to be UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize(value: Any) -> str:
    return str(value).strip().upper()


def _excerpt(text: str, *, max_chars: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3].rstrip()}..."


_STOPWORDS = {
    "about",
    "and",
    "are",
    "does",
    "did",
    "for",
    "from",
    "have",
    "how",
    "into",
    "its",
    "over",
    "say",
    "says",
    "the",
    "their",
    "what",
    "with",
}
_SAFE_HARBOR_TERMS = ("safe harbor", "forward-looking", "undue reliance", "actual results may differ")


def _content_terms(value: str) -> list[str]:
    terms = re.findall(r"[a-z0-9][a-z0-9-]{1,}", value.lower())
    return [term for term in terms if term not in _STOPWORDS]


def _query_phrases(query: str) -> tuple[str, ...]:
    lower = query.lower()
    phrases = []
    for phrase in (
        "item 1a",
        "risk factors",
        "export controls",
        "data center",
        "gross margin",
        "supply chain",
        "press release",
        "cfo commentary",
        "capital allocation",
        "interest rate",
        "energy transition",
    ):
        if phrase in lower:
            phrases.append(phrase)
    return tuple(phrases)


def _metadata_relevance_bonus(query: str, metadata: dict[str, Any]) -> float:
    lower = query.lower()
    bonus = 0.0
    if ("item 1a" in lower or "risk factor" in lower) and _normalize(metadata.get("item_number")) == "1A":
        bonus += 0.9
    if "cfo" in lower and _normalize(metadata.get("exhibit_type")) == "CFO_COMMENTARY":
        bonus += 0.8
    if "press release" in lower and _normalize(metadata.get("exhibit_type")) == "PRESS_RELEASE":
        bonus += 0.8
    if "risk" in lower and _normalize(metadata.get("item_number")) == "1A":
        bonus += 0.35
    return bonus


def _domain_phrase_bonus(query: str, text_lower: str) -> float:
    lower = query.lower()
    bonus = 0.0
    if _asks_about_capital_return(lower):
        if any(
            term in text_lower
            for term in ("repurchase", "repurchased", "buyback", "buybacks", "issuer purchases")
        ):
            bonus += 1.0
        elif any(term in text_lower for term in ("dividend", "shareholders")):
            bonus += 0.5
    if "demand" in lower and "demand" in text_lower:
        bonus += 0.5
    if "inventory" in lower and "inventory" in text_lower:
        bonus += 0.5
    return bonus


def _asks_about_capital_return(lower_query: str) -> bool:
    """Detect capital-return phrasing, including buyback/repurchase wording.

    The original bonus only fired on "capital allocation" / "shareholder
    returns"; queries phrased as "capital return or buybacks" (common for banks
    and energy issuers) got no repurchase boost, so issuer-purchase evidence
    ranked below generic financial-highlights chunks.
    """

    return any(
        term in lower_query
        for term in (
            "capital allocation",
            "shareholder returns",
            "capital return",
            "buyback",
            "buybacks",
            "share repurchase",
            "share repurchases",
            "repurchase",
        )
    )


def _asks_for_operating_risks(query: str) -> bool:
    lower = query.lower()
    return "risk" in lower and "safe harbor" not in lower


def _is_safe_harbor_only(text: str) -> bool:
    lower = text.lower()
    if not any(term in lower for term in _SAFE_HARBOR_TERMS):
        return False
    return not any(
        term in lower
        for term in (
            "supply",
            "export",
            "competition",
            "credit",
            "commodity",
            "margin",
            "data center",
            "inventory",
            "cybersecurity",
            "regulation",
        )
    )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)
