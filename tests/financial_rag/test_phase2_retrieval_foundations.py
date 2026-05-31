from src.financial_rag.chunking import chunk_document
from src.financial_rag.evaluation import (
    RetrievalEvalCase,
    evaluate_retrieval_results,
    mean_reciprocal_rank,
    recall_at_k,
)
from src.financial_rag.ingestion.sec_client import infer_exhibit_type
from src.financial_rag.models import FilingMetadata
from src.financial_rag.parsing import classify_ex99_exhibit, parse_sec_sections
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever, RetrievalFilters
from src.financial_rag.synthesis import extract_citation_labels, validate_citations


def test_tenk_and_tenq_item_boundary_parser_preserves_item_metadata() -> None:
    text = """
    Item 1. Business
    We sell accelerated computing products.

    Item 1A. Risk Factors
    Demand, supply, regulation, and competition may affect results.

    Item 7. Management's Discussion and Analysis
    Revenue changed because customers bought more data center products.
    """

    sections = parse_sec_sections(text, form_type="10-K")

    assert [section.item_number for section in sections] == ["1", "1A", "7"]
    assert sections[1].section_path == "Item 1A. Risk Factors"
    assert "regulation" in sections[1].text


def test_8k_item_boundary_parser() -> None:
    text = """
    Item 2.02 Results of Operations and Financial Condition
    The company issued quarterly results.

    Item 9.01 Financial Statements and Exhibits
    Exhibit 99.1 is furnished.
    """

    sections = parse_sec_sections(text, form_type="8-K")

    assert [section.item_number for section in sections] == ["2.02", "9.01"]
    assert sections[0].section_path.startswith("Item 2.02.")


def test_ex99_classification_uses_filename_description_and_text_hints() -> None:
    assert infer_exhibit_type("q1fy27cfocommentary.htm", "EX-99.2") == "CFO_COMMENTARY"
    assert infer_exhibit_type("amdq126earningsslidesfin.htm", "EX-99.2") == "PRESENTATION"
    assert infer_exhibit_type("pressreleasedatedfebruary2.htm", "EX-99.1") == "PRESS_RELEASE"
    assert (
        classify_ex99_exhibit(
            filename="ex991.htm",
            description="EX-99.1",
            text="Prepared Remarks\nJensen Huang: Welcome to our call.",
        )
        == "PREPARED_REMARKS"
    )
    assert classify_ex99_exhibit(filename="ex991.htm", description="EX-99.1") == "EX-99"


def test_item_aware_chunks_are_stable_and_keep_section_metadata() -> None:
    metadata = _metadata(form_type="10-K")
    text = (
        "Item 1. Business\n"
        + ("Business overview text. " * 40)
        + "\n\nItem 1A. Risk Factors\n"
        + ("Risk factor text. Export controls and demand shifts may affect revenue. " * 40)
    )

    first = chunk_document(text, metadata, max_chars=260, overlap_chars=0)
    second = chunk_document(text, metadata, max_chars=260, overlap_chars=0)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert any(chunk.item_number == "1A" for chunk in first)
    risk_chunk = next(chunk for chunk in first if chunk.item_number == "1A")
    assert risk_chunk.section_path == "Item 1A. Risk Factors"
    assert risk_chunk.metadata["item_number"] == "1A"


def test_section_parser_falls_back_when_items_only_in_trailing_index() -> None:
    # Mimics Intel's 10-K: the body uses bare headings while literal "Item N."
    # labels appear only in a trailing cross-reference index. Building sections
    # from those labels would drop the entire body before the first match.
    body = "Risk Factors\n" + ("Supply constraints and demand shifts affect results. " * 100)
    trailing_index = (
        "\nItem 1. Business: 5\n"
        "Item 1A. Risk Factors 37\n"
        "Item 7. Management's Discussion and Analysis of Financial Condition 50\n"
    )

    sections = parse_sec_sections(body + trailing_index, form_type="10-K")

    assert len(sections) == 1
    assert sections[0].item_number == ""  # fell back to a whole-document section
    assert "Supply constraints and demand shifts" in sections[0].text


def test_chunking_preserves_body_when_item_labels_only_in_trailing_index() -> None:
    metadata = _metadata(form_type="10-K")
    body = (
        "Risk Factors\n"
        + ("Our manufacturing and foundry operations face supply constraints. " * 60)
        + "\nData center demand and competition affect revenue growth. " * 60
    )
    text = body + "\nItem 1. Business: 5\nItem 1A. Risk Factors 37\n"

    chunks = chunk_document(text, metadata, max_chars=400, overlap_chars=0)

    joined = " ".join(chunk.chunk_text for chunk in chunks)
    assert "manufacturing and foundry operations" in joined
    assert "Data center demand and competition" in joined
    # The body must actually chunk, not collapse to a handful of index lines.
    assert len(chunks) >= 5


def test_ex99_speaker_chunks_only_when_obvious_labels_exist() -> None:
    metadata = _metadata(form_type="EX-99", document_role="exhibit", exhibit_type="PREPARED_REMARKS")
    text = """
    Jensen Huang: Data center demand remains strong.

    Colette Kress: Gross margin improved due to product mix.
    """

    chunks = chunk_document(text, metadata, max_chars=500, overlap_chars=0)

    assert [chunk.speaker_name for chunk in chunks] == ["Jensen Huang", "Colette Kress"]
    assert chunks[0].section_path == "Speaker: Jensen Huang"


def test_fake_vector_retrieval_filters_and_scores() -> None:
    retriever = LocalDenseRetriever(
        chunks=[
            _chunk("a", "risk text", ticker="NVDA", form_type="10-K", item_number="1A"),
            _chunk("b", "revenue text", ticker="NVDA", form_type="10-Q", item_number="2"),
            _chunk("c", "other company", ticker="AMD", form_type="10-K", item_number="1A"),
        ],
        embeddings={"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 0.0]},
    )

    results = retriever.search(
        query_vector=[1.0, 0.0],
        filters=RetrievalFilters(ticker="NVDA", form_type="10-K", item_number="1A"),
        top_k=5,
    )

    assert [result.chunk_id for result in results] == ["a"]
    assert results[0].dense_score == 1.0
    assert results[0].citation_label == "S1"


def test_citation_validation_rejects_hallucinated_labels_and_hydrates_metadata() -> None:
    retriever = LocalDenseRetriever(
        chunks=[_chunk("a", "risk text", ticker="NVDA", form_type="10-K", item_number="1A")],
        embeddings={"a": [1.0]},
    )
    results = retriever.search(query_vector=[1.0], top_k=1)
    labels = extract_citation_labels("NVIDIA describes risks [S1] and unsupported detail [S9].")

    validation = validate_citations(labels, results)

    assert [citation.label for citation in validation.accepted] == ["S1"]
    assert validation.accepted[0].ticker == "NVDA"
    assert validation.accepted[0].form_type == "10-K"
    assert validation.accepted[0].chunk_id == "a"
    assert validation.rejected == ["S9"]


def test_retrieval_eval_metrics() -> None:
    retrieved = ["a", "b", "c"]

    assert recall_at_k(retrieved, {"b", "z"}, k=2) == 0.5
    results = evaluate_retrieval_results(
        [
            RetrievalEvalCase(query_id="q1", question="Relevant at rank 2.", relevant_chunk_ids={"b"}),
            RetrievalEvalCase(query_id="q2", question="No relevant result.", relevant_chunk_ids={"z"}),
        ],
        {"q1": retrieved, "q2": retrieved},
        k=3,
    )
    assert results[0].reciprocal_rank == 0.5
    assert results[1].recall_at_k == 0.0
    assert mean_reciprocal_rank([result.reciprocal_rank for result in results]) == 0.25


def _metadata(
    *,
    form_type: str,
    document_role: str = "primary",
    exhibit_type: str = "",
) -> FilingMetadata:
    return FilingMetadata(
        document_id=f"NVDA-{form_type}-doc",
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA CORP",
        accession_number="0001045810-26-000021",
        form_type=form_type,
        filing_date="2026-02-25",
        report_date="2026-01-25",
        source_url="https://www.sec.gov/Archives/doc.htm",
        local_path="data/filings/raw/NVDA/doc.htm",
        document_role=document_role,
        exhibit_type=exhibit_type,
        document_name="doc.htm",
        description=form_type,
        content_hash="abc123",
    )


def _chunk(
    chunk_id: str,
    text: str,
    *,
    ticker: str,
    form_type: str,
    item_number: str,
) -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "doc",
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": "2026-02-25",
            "accession_number": "0001045810-26-000021",
            "source_url": "https://www.sec.gov/Archives/doc.htm",
            "document_role": "primary",
            "exhibit_type": "",
            "item_number": item_number,
        },
    )
