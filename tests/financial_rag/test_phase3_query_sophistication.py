from src.financial_rag.evaluation import PHASE3_ROUTED_RETRIEVAL_FIXTURES, evaluate_retrieval_results
from src.financial_rag.query import (
    QueryPipeline,
    build_coverage_report,
    hydrate_parent_context,
    plan_retrieval,
    route_query,
)
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever, RetrievalResult
from src.financial_rag.scope import QueryType


def test_rule_based_router_extracts_temporal_filters() -> None:
    routed = route_query(
        "How have NVIDIA risk disclosures changed over the last year?",
        default_ticker="NVDA",
    )

    assert routed.query_type == QueryType.TEMPORAL
    assert routed.filters.tickers == ["NVDA"]
    assert routed.filters.time_window == "last_year"
    assert routed.filters.last_n_quarters == 4


def test_router_extracts_cross_company_and_item_filters() -> None:
    routed = route_query("Compare NVDA and AMD Item 1A risk factors.")

    assert routed.query_type == QueryType.CROSS_COMPANY
    assert routed.filters.tickers == ["NVDA", "AMD"]
    assert routed.filters.item_numbers == ["1A"]
    assert routed.filters.document_roles == ["primary"]


def test_router_forces_risk_factor_and_exhibit_filters() -> None:
    risk = route_query("What risk factors does NVIDIA describe?")
    cfo = route_query("What did NVIDIA CFO commentary say about revenue?")
    press_release = route_query("What does NVIDIA's press release say about gross margin?")

    assert risk.filters.form_types == ["10-K", "10-Q"]
    assert risk.filters.item_numbers == ["1A"]
    assert risk.filters.document_roles == ["primary"]
    assert cfo.filters.form_types == ["EX-99"]
    assert cfo.filters.exhibit_types == ["CFO_COMMENTARY"]
    assert press_release.filters.form_types == ["EX-99"]
    assert press_release.filters.exhibit_types == ["PRESS_RELEASE"]


def test_planner_decomposes_temporal_and_cross_company_queries() -> None:
    temporal = plan_retrieval(route_query("How did NVDA revenue change over the last 3 quarters?"))
    cross_company = plan_retrieval(route_query("Compare NVDA and AMD data center commentary."))

    assert [subquery.subquery_id for subquery in temporal] == [
        "temporal-01",
        "temporal-02",
        "temporal-03",
    ]
    assert [subquery.filters.ticker for subquery in cross_company] == ["NVDA", "AMD"]


def test_planner_preserves_speaker_and_document_role_filters() -> None:
    subqueries = plan_retrieval(route_query("What did NVIDIA CFO commentary say about revenue?"))

    assert len(subqueries) == 1
    assert subqueries[0].filters.ticker == "NVDA"
    assert subqueries[0].filters.document_role == "exhibit"
    assert subqueries[0].filters.exhibit_type == "CFO_COMMENTARY"
    assert subqueries[0].filters.form_types == ("EX-99",)
    assert subqueries[0].filters.speaker_role == "CFO"


def test_parent_context_hydrates_nearby_same_document_chunks() -> None:
    chunks = [
        _chunk("a", "before", document_id="doc", start=0),
        _chunk("b", "target", document_id="doc", start=100),
        _chunk("c", "after", document_id="doc", start=200),
        _chunk("d", "other", document_id="other", start=0),
    ]
    result = RetrievalResult(
        chunk_id="b",
        rank=1,
        dense_score=1.0,
        citation_label="S1",
        source_url="https://www.sec.gov/doc",
        source_excerpt="target",
        metadata=chunks[1].metadata,
    )

    hydrated = hydrate_parent_context(result, chunks, window=1)

    assert hydrated.context_chunk_ids == ["a", "b", "c"]
    assert "before" in hydrated.context_text
    assert "after" in hydrated.context_text
    assert "other" not in hydrated.context_text


def test_coverage_report_surfaces_cached_roles_and_gaps() -> None:
    report = build_coverage_report(
        [
            _chunk("a", "risk", form_type="10-K", document_role="primary", item_number="1A"),
            _chunk(
                "b",
                "cfo",
                form_type="EX-99",
                document_role="exhibit",
                exhibit_type="CFO_COMMENTARY",
                speaker_name="Colette Kress",
            ),
        ],
        tickers=["NVDA", "AMD"],
    )

    nvda = report.tickers["NVDA"]
    amd = report.tickers["AMD"]
    assert nvda.ex99_chunk_count == 1
    assert nvda.has_cfo_commentary is True
    assert nvda.has_press_release is False
    assert nvda.form_types == ["10-K", "EX-99"]
    assert nvda.exhibit_types == ["CFO_COMMENTARY"]
    assert nvda.item_numbers == ["1A"]
    assert nvda.speakers == ["Colette Kress"]
    assert "No local chunks cached." in amd.gaps
    assert "No cached EX-99 exhibits detected." in amd.gaps


def test_query_pipeline_orchestrates_fake_vector_retrieval_and_citations() -> None:
    chunks = [
        _chunk("risk", "risk disclosure", form_type="10-K", item_number="1A"),
        _chunk(
            "cfo",
            "cfo commentary revenue",
            form_type="EX-99",
            document_role="exhibit",
            exhibit_type="CFO_COMMENTARY",
            speaker_name="Colette Kress",
        ),
        _chunk("amd", "amd commentary", ticker="AMD", form_type="10-K"),
    ]
    retriever = LocalDenseRetriever(
        chunks=chunks,
        embeddings={"risk": [1.0, 0.0], "cfo": [0.9, 0.1], "amd": [0.0, 1.0]},
        query_embedder=_FakeEmbedder(),
    )

    result = QueryPipeline(retriever=retriever, chunks=chunks).run(
        "What does NVDA Item 1A say about risk?",
        top_k=2,
    )

    assert result.routed_query.query_type == QueryType.SINGLE_DOC_LOOKUP
    assert [item.citation_label for item in result.results] == ["S1"]
    assert result.results[0].metadata["item_number"] == "1A"
    assert result.citation_validation.rejected == []
    assert result.citation_validation.accepted[0].chunk_id == "risk"


def test_phase3_eval_fixtures_and_metrics_remain_offline() -> None:
    assert {case.expected_query_type for case in PHASE3_ROUTED_RETRIEVAL_FIXTURES} >= {
        "temporal",
        "cross_company",
        "speaker_specific",
    }
    results = evaluate_retrieval_results(
        [PHASE3_ROUTED_RETRIEVAL_FIXTURES[0]],
        {PHASE3_ROUTED_RETRIEVAL_FIXTURES[0].query_id: ["a"]},
        k=5,
    )
    assert results[0].recall_at_k == 0.0


class _FakeEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "amd" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors


def _chunk(
    chunk_id: str,
    text: str,
    *,
    ticker: str = "NVDA",
    document_id: str = "doc",
    form_type: str = "10-K",
    document_role: str = "primary",
    exhibit_type: str = "",
    item_number: str = "",
    speaker_name: str = "",
    speaker_role: str = "",
    start: int = 0,
) -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": "2026-02-25",
            "accession_number": "0001045810-26-000021",
            "source_url": "https://www.sec.gov/Archives/doc.htm",
            "document_role": document_role,
            "exhibit_type": exhibit_type,
            "item_number": item_number,
            "speaker_name": speaker_name,
            "speaker_role": speaker_role,
            "start_offset": start,
            "end_offset": start + len(text),
        },
    )
