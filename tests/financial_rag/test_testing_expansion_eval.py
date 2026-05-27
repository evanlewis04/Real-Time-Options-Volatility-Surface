import importlib
import json
from pathlib import Path

from src.financial_rag.evaluation import (
    EXPANDED_ANSWER_CASES,
    EXPANDED_RETRIEVAL_CASES,
    ExpandedEvalCase,
    SourceConstraint,
    apply_gold_labels_to_cases,
    build_answer_quality_report,
    build_retrieval_quality_report,
    count_uncited_factual_sentences,
    evaluate_answer_case,
    evaluate_retrieval_case,
    filter_cases,
    gold_label_summary,
    resolve_gold_labels,
    write_csv_rows,
    write_json_report,
)
from src.financial_rag.retrieval import LocalChunkRecord, LocalDenseRetriever
from src.financial_rag.synthesis import synthesize_answer_from_query_payload


def test_expanded_fixture_scale_and_company_coverage() -> None:
    companies = {ticker for case in EXPANDED_RETRIEVAL_CASES for ticker in case.tickers}

    assert 25 <= len(EXPANDED_RETRIEVAL_CASES) <= 50
    assert 10 <= len(EXPANDED_ANSWER_CASES) <= 25
    assert {"NVDA", "AMD", "MSFT", "AAPL", "JPM", "XOM"} <= companies
    assert filter_cases(EXPANDED_RETRIEVAL_CASES, tickers=["NVDA"], max_cases=3)


def test_source_constraints_and_retrieval_metrics_pass_with_matching_payload() -> None:
    case = ExpandedEvalCase(
        "fixture-risk",
        "What risks does NVIDIA describe?",
        ("NVDA",),
        "single_doc_lookup",
        (
            SourceConstraint(
                "NVDA",
                form_type="10-K",
                item_number="1A",
                required_keywords=("export",),
                forbidden_keywords=("safe harbor",),
            ),
        ),
        relevant_chunk_ids=frozenset({"risk"}),
    )

    result = evaluate_retrieval_case(case, _query_payload(), k=5)

    assert result.status == "pass"
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.source_hit is True
    assert result.metadata_complete is True


def test_source_constraints_use_parent_context_and_keyword_aliases() -> None:
    case = ExpandedEvalCase(
        "fixture-capital",
        "Capital return?",
        ("JPM",),
        "single_doc_lookup",
        (SourceConstraint("JPM", required_keywords=("capital", "repurchase")),),
    )
    payload = {
        "results": [
            {
                "chunk_id": "capital",
                "citation_label": "S1",
                "source_url": "https://www.sec.gov/Archives/capital.htm",
                "source_excerpt": "Capital planning discussion.",
                "metadata": {
                    "ticker": "JPM",
                    "form_type": "10-K",
                    "filing_date": "2026-02-01",
                    "accession_number": "capital-accession",
                },
                "parent_context": {"context_text": "The firm returned capital through buybacks and dividends."},
            }
        ],
        "citations": {"accepted": [{"label": "S1"}], "rejected": []},
    }

    assert evaluate_retrieval_case(case, payload).source_hit is True


def test_retrieval_failure_classification_for_wrong_section_safe_harbor_and_metadata() -> None:
    case = ExpandedEvalCase(
        "fixture-fail",
        "What operating risks does NVIDIA describe?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA", form_type="10-K", item_number="1A", required_keywords=("export",)),),
    )
    payload = {
        "results": [
            {
                "chunk_id": "safe",
                "citation_label": "S1",
                "source_url": "",
                "source_excerpt": "Safe harbor forward-looking statements actual results may differ.",
                "metadata": {"ticker": "NVDA", "form_type": "10-K"},
                "parent_context": None,
            }
        ],
        "citations": {"accepted": [], "rejected": []},
    }

    result = evaluate_retrieval_case(case, payload)

    assert result.status == "fail"
    assert "wrong_section_or_source" in result.failures
    assert "missing_metadata" in result.failures
    assert "safe_harbor_only" in result.failures


def test_retrieval_report_and_writers(tmp_path: Path) -> None:
    case = ExpandedEvalCase(
        "fixture-risk",
        "Risk?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA", form_type="10-K", required_keywords=("export",)),),
    )

    report = build_retrieval_quality_report([case], {"fixture-risk": _query_payload()})
    json_path = write_json_report(report, tmp_path / "artifacts" / "rag_eval" / "retrieval.json")
    csv_path = write_csv_rows(report["results"], tmp_path / "artifacts" / "rag_eval" / "retrieval.csv")

    assert report["case_count"] == 1
    assert report["section_source_hit_rate"] == 1.0
    assert json.loads(json_path.read_text(encoding="utf-8"))["case_count"] == 1
    assert csv_path.exists()


def test_gold_recall_is_case_level_when_multiple_chunks_are_relevant() -> None:
    case = ExpandedEvalCase(
        "fixture-risk",
        "Risk?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA", form_type="10-K", required_keywords=("export",)),),
        relevant_chunk_ids=frozenset({"risk", "alternate-risk"}),
    )

    result = evaluate_retrieval_case(case, _query_payload(), k=5)

    assert result.recall_at_k == 1.0


def test_answer_eval_flags_hallucinated_and_uncited_sentences() -> None:
    case = ExpandedEvalCase(
        "fixture-answer",
        "Risk?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA"),),
    )
    answer = {
        "answer_text": "NVIDIA mentions export restrictions [S1]. This sentence has a factual claim without a citation.",
        "accepted_citations": [{"label": "S1"}],
        "rejected_citations": ["S9"],
        "dry_run": False,
    }

    result = evaluate_answer_case(case, answer)

    assert result.status == "warning"
    assert result.rejected_citation_count == 1
    assert result.uncited_sentence_count == 1
    assert "hallucinated_citation" in result.failures
    assert count_uncited_factual_sentences(answer["answer_text"]) == 1


def test_answer_report_and_dry_run_answer_eval() -> None:
    case = ExpandedEvalCase(
        "fixture-answer",
        "Risk?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA"),),
    )
    answer = synthesize_answer_from_query_payload(_query_payload(), question=case.question, dry_run=True)
    report = build_answer_quality_report([case], {"fixture-answer": answer})

    assert answer.accepted_citations
    assert report["case_count"] == 1
    assert report["hallucinated_citation_count"] == 0
    assert report["pass_rate"] == 1.0


def test_expanded_eval_scripts_importable_and_phase_commands_compatible() -> None:
    modules = [
        "scripts.financial_rag_expanded_retrieval_eval",
        "scripts.financial_rag_expanded_answer_eval",
        "scripts.financial_rag_phase1_smoke",
        "scripts.financial_rag_phase7_api_smoke",
        "scripts.financial_rag_openai_answer_smoke",
        "scripts.financial_rag_retrieval_repair",
    ]

    for module in modules:
        assert importlib.import_module(module)


def test_gold_labels_resolve_to_current_chunk_ids() -> None:
    chunks = [
        _chunk_record(
            "nvda-risk",
            "Item 1A risk factors discuss export controls and supply constraints.",
            ticker="NVDA",
            form_type="10-K",
            item_number="1A",
        ),
        _chunk_record(
            "amd-risk",
            "Item 1A risk factors discuss supply availability.",
            ticker="AMD",
            form_type="10-K",
            item_number="1A",
        ),
    ]

    labels = resolve_gold_labels(chunks)
    labeled_cases = apply_gold_labels_to_cases(EXPANDED_RETRIEVAL_CASES, labels)
    nvda_case = next(case for case in labeled_cases if case.case_id == "nvda-item1a-supply")

    assert "nvda-risk" in nvda_case.relevant_chunk_ids
    assert gold_label_summary(labels)["label_count"] >= 2


def test_local_retrieval_suppresses_safe_harbor_for_risk_queries() -> None:
    chunks = [
        _chunk_record(
            "safe",
            "Safe harbor forward-looking statements actual results may differ.",
            item_number="",
        ),
        _chunk_record(
            "risk",
            "Item 1A risk factors include export controls and supply constraints.",
            item_number="1A",
        ),
    ]
    retriever = LocalDenseRetriever(chunks=chunks, embeddings={}, query_embedder=_ConstantEmbedder())

    results = retriever.search(query="What risk factors does NVIDIA describe?", top_k=2)

    assert results[0].chunk_id == "risk"


def _query_payload() -> dict[str, object]:
    return {
        "results": [
            {
                "chunk_id": "risk",
                "citation_label": "S1",
                "source_url": "https://www.sec.gov/Archives/risk.htm",
                "source_excerpt": "Item 1A risk factors discuss export controls and supply constraints.",
                "metadata": {
                    "ticker": "NVDA",
                    "form_type": "10-K",
                    "filing_date": "2026-02-01",
                    "accession_number": "risk-accession",
                    "item_number": "1A",
                    "document_role": "primary",
                },
                "parent_context": {"context_chunk_ids": ["risk"]},
            }
        ],
        "citations": {
            "accepted": [
                {
                    "label": "S1",
                    "ticker": "NVDA",
                    "form_type": "10-K",
                    "filing_date": "2026-02-01",
                    "accession": "risk-accession",
                    "source_url": "https://www.sec.gov/Archives/risk.htm",
                    "chunk_id": "risk",
                }
            ],
            "rejected": [],
        },
    }


class _ConstantEmbedder:
    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _text in texts]


def _chunk_record(
    chunk_id: str,
    text: str,
    *,
    ticker: str = "NVDA",
    form_type: str = "10-K",
    item_number: str = "",
    document_role: str = "primary",
    exhibit_type: str = "",
) -> LocalChunkRecord:
    return LocalChunkRecord(
        chunk_id=chunk_id,
        chunk_text=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": f"doc-{chunk_id}",
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": "2026-02-01",
            "accession_number": f"accession-{chunk_id}",
            "source_url": f"https://www.sec.gov/Archives/{chunk_id}.htm",
            "document_role": document_role,
            "exhibit_type": exhibit_type,
            "item_number": item_number,
            "section_path": f"Item {item_number}" if item_number else "",
        },
    )
