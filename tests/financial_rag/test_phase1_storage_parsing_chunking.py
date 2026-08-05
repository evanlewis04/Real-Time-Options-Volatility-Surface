from pathlib import Path

from src.financial_rag.chunking import CHUNKER_VERSION, chunk_document
from src.financial_rag.models import FilingMetadata
from src.financial_rag.parsing import PARSER_VERSION, extract_readable_text
from src.financial_rag.storage import LocalRagStore


def test_local_store_uses_ignored_phase1_paths_and_idempotent_manifests(tmp_path: Path) -> None:
    store = LocalRagStore(root=tmp_path)
    raw_path = store.raw_path("NVDA", "0001045810-26-000051", "nvda.htm")

    first = store.write_bytes_once(raw_path, b"first")
    second = store.write_bytes_once(raw_path, b"second")
    changed_first = store.upsert_manifest(
        store.raw_dir / "manifest.jsonl",
        key="document_id",
        record={"document_id": "doc-1", "source_url": "https://www.sec.gov/a"},
    )
    changed_second = store.upsert_manifest(
        store.raw_dir / "manifest.jsonl",
        key="document_id",
        record={"document_id": "doc-1", "source_url": "https://www.sec.gov/a"},
    )

    assert first.created is True
    assert second.created is False
    assert raw_path.read_bytes() == b"first"
    assert store.raw_dir == tmp_path / "data" / "filings" / "raw"
    assert store.parsed_dir == tmp_path / "data" / "filings" / "parsed"
    assert store.chunks_dir == tmp_path / "data" / "filings" / "chunks"
    assert store.vector_cache_dir == tmp_path / "data" / "vector_cache"
    assert changed_first is True
    assert changed_second is False


def test_append_manifest_appends_o1_without_rewriting_existing_rows(tmp_path: Path) -> None:
    """append_manifest adds one row without rereading/rewriting the file.

    This is the hot-path primitive behind the embedding- and chunk-manifest writes:
    the old per-row `upsert_manifest` rewrote the whole (tens-to-hundreds of MB)
    manifest each call, which is O(n^2) over an ingest run. append_manifest must
    leave every pre-existing byte intact and only append.
    """
    store = LocalRagStore(root=tmp_path)
    manifest_path = store.chunks_dir / "manifest.jsonl"
    original = '{"chunk_id": "pre", "note": "keep me verbatim"}\n'
    manifest_path.write_text(original, encoding="utf-8")

    store.append_manifest(manifest_path, {"chunk_id": "new-1", "n": 1})
    store.append_manifest(manifest_path, {"chunk_id": "new-2", "n": 2})

    content = manifest_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert content.startswith(original)  # pre-existing bytes untouched
    assert len(lines) == 3
    assert lines[1] == '{"chunk_id": "new-1", "n": 1}'  # sorted-keys, one appended row
    assert lines[2] == '{"chunk_id": "new-2", "n": 2}'


def test_minimal_html_extraction_removes_scripts_and_normalizes_whitespace() -> None:
    html = """
    <html><head><style>.x { color: red; }</style><script>bad()</script></head>
    <body><h1>Risk Factors</h1><p>Revenue&nbsp;may vary.</p><p>Demand can shift.</p></body></html>
    """

    text = extract_readable_text(html)

    assert "bad()" not in text
    assert "Risk Factors" in text
    assert "Revenue may vary." in text
    assert "\n\n\n" not in text


def test_chunk_ids_and_metadata_are_stable() -> None:
    metadata = FilingMetadata(
        document_id="NVDA-doc",
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA CORP",
        accession_number="0001045810-26-000051",
        form_type="10-Q",
        filing_date="2026-05-20",
        report_date="2026-04-26",
        source_url="https://www.sec.gov/Archives/doc.htm",
        local_path="data/filings/raw/NVDA/doc.htm",
        document_role="primary",
        exhibit_type="",
        document_name="doc.htm",
        description="10-Q",
        content_hash="abc123",
    )
    text = "Alpha beta gamma. " * 200

    first = chunk_document(text, metadata, max_chars=300, overlap_chars=20)
    second = chunk_document(text, metadata, max_chars=300, overlap_chars=20)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].ticker == "NVDA"
    assert first[0].accession_number == "0001045810-26-000051"
    assert first[0].metadata["parser_version"] == PARSER_VERSION
    assert first[0].metadata["chunker_version"] == CHUNKER_VERSION
    assert first[0].metadata["content_hash"] == "abc123"
