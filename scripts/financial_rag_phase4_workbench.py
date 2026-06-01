"""Streamlit filings intelligence workbench for local Phase 4 evidence review."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.financial_rag.api import LocalApiError, LocalRagApiService, QueryRequest, build_local_api_service
from src.financial_rag.audit import build_evidence_quality_report, build_readiness_report
from src.financial_rag.settings import project_root
from src.financial_rag.synthesis import check_openai_readiness, synthesize_answer_from_query_payload
from src.financial_rag.workbench import (
    answer_citation_rows,
    change_rows,
    company_options,
    coverage_rows,
    evaluate_answer_gate,
    evidence_quality_issue_rows,
    evidence_rows,
    language_signal_rows,
    readiness_issue_rows,
    rejected_citation_rows,
)


def main() -> None:
    st.set_page_config(page_title="Filings Intelligence Workbench", layout="wide")
    st.title("Filings Intelligence Workbench")
    st.caption(
        "Local SEC filings evidence review. Answers are opt-in, OpenAI-only, and "
        "gated behind retrieval quality; no SEC refetch."
    )

    with st.sidebar:
        st.header("Query")
        use_voyage = st.checkbox("Use Voyage query embeddings", value=True)
        top_k = st.number_input("Top-k", min_value=1, max_value=20, value=5, step=1)
        per_subquery_k = st.number_input("Per-subquery k", min_value=1, max_value=20, value=5, step=1)

    service = build_local_api_service(root=project_root(), use_voyage=use_voyage)
    options = company_options(service.companies()) or ["NVDA"]
    default_index = options.index("NVDA") if "NVDA" in options else 0
    with st.sidebar:
        ticker = st.selectbox("Ticker", options, index=default_index)

    health = service.health()
    st.info(f"Local cache: {health['chunk_count']} chunks, {health['embedding_count']} embeddings")

    readiness = build_readiness_report(
        service.chunks,
        service.retriever.embeddings,
        tickers=[ticker],
        root=project_root(),
    )
    with st.expander("Phase 6 Readiness"):
        left, middle, right = st.columns(3)
        left.metric("Status", readiness.status)
        middle.metric("Missing Embeddings", readiness.missing_embedding_count)
        right.metric("Missing Item Metadata", readiness.missing_item_metadata_count)
        st.dataframe(readiness_issue_rows(readiness.to_dict()), use_container_width=True)
        st.json(
            {
                "unsupported_tickers": readiness.unsupported_tickers,
                "ex99_coverage": readiness.ex99_coverage,
                "companyfacts_available": readiness.companyfacts_available,
            }
        )

    question = st.text_area(
        "Question",
        value="How have NVIDIA risk disclosures changed over the last year?",
        height=90,
    )

    if st.button("Retrieve Evidence", type="primary"):
        payload = _retrieve(service, question=question, ticker=ticker, top_k=top_k, per_subquery_k=per_subquery_k)
        if payload is not None:
            _render_evidence(payload)

    _render_answer_section(
        service,
        question=question,
        ticker=ticker,
        top_k=top_k,
        per_subquery_k=per_subquery_k,
    )

    with st.expander("Coverage Only"):
        st.dataframe(coverage_rows(service.coverage(tickers=[ticker])), use_container_width=True)

    with st.expander("Phase 5 Differentiators"):
        differentiators = service.differentiators(ticker=ticker)
        st.write("Filing Changes")
        st.dataframe(change_rows(differentiators), use_container_width=True)
        st.write("Language Signals")
        st.dataframe(language_signal_rows(differentiators), use_container_width=True)
        st.write("XBRL")
        st.json(differentiators["xbrl"])
        st.write("Market Context")
        st.json(differentiators["market_context"])


def _retrieve(
    service: LocalRagApiService,
    *,
    question: str,
    ticker: str,
    top_k: int,
    per_subquery_k: int,
) -> dict[str, Any] | None:
    try:
        return service.query(
            QueryRequest(
                question=question,
                ticker=ticker,
                top_k=int(top_k),
                per_subquery_k=int(per_subquery_k),
            )
        )
    except LocalApiError as exc:
        st.error(f"{exc.code}: {exc.message}")
        st.json(exc.to_dict())
        return None


def _render_evidence(payload: dict[str, Any]) -> None:
    st.subheader("Routing")
    st.json(payload["routed_query"])

    st.subheader("Subqueries")
    st.dataframe(payload["subqueries"], use_container_width=True)

    st.subheader("Evidence")
    st.dataframe(evidence_rows(payload), use_container_width=True)
    for result in payload["results"]:
        with st.expander(f"{result['citation_label']} {result['chunk_id']}"):
            st.write(result["source_excerpt"])
            st.json(result["metadata"])
            st.write(result.get("parent_context", {}).get("context_text", ""))

    st.subheader("Citations")
    st.json(payload["citations"])
    rejected = rejected_citation_rows(payload)
    if rejected:
        st.dataframe(rejected, use_container_width=True)

    st.subheader("Evidence Quality")
    evidence_quality = build_evidence_quality_report(payload)
    q_left, q_middle, q_right = st.columns(3)
    q_left.metric("Status", evidence_quality.status)
    q_middle.metric("Missing URLs", evidence_quality.missing_url_count)
    q_right.metric("Invalid Citations", evidence_quality.invalid_citation_count)
    st.dataframe(evidence_quality_issue_rows(evidence_quality.to_dict()), use_container_width=True)

    st.subheader("Coverage")
    st.dataframe(coverage_rows(payload["coverage"]), use_container_width=True)


def _render_answer_section(
    service: LocalRagApiService,
    *,
    question: str,
    ticker: str,
    top_k: int,
    per_subquery_k: int,
) -> None:
    st.subheader("Grounded Answer (opt-in)")
    openai_readiness = check_openai_readiness()
    st.caption(f"OpenAI: {openai_readiness.status} (model {openai_readiness.model})")
    for issue in openai_readiness.issues:
        st.write(f"- {issue}")

    preview_col, generate_col = st.columns(2)
    run_preview = preview_col.button("Preview Answer Prompt (dry run, no cost)")
    run_generate = generate_col.button("Generate Grounded Answer (OpenAI)")

    if not (run_preview or run_generate):
        return

    payload = _retrieve(service, question=question, ticker=ticker, top_k=top_k, per_subquery_k=per_subquery_k)
    if payload is None:
        return

    if run_preview:
        answer = synthesize_answer_from_query_payload(payload, question=question, dry_run=True)
        st.write(answer.answer_text)
        st.caption(f"Retrieved labels: {', '.join(answer.retrieved_labels) or 'none'}")
        with st.expander("Prompt preview"):
            st.code(answer.prompt_preview)
        return

    evidence_quality = build_evidence_quality_report(payload)
    gate = evaluate_answer_gate(
        payload,
        evidence_quality_status=evidence_quality.status,
        openai_ready=openai_readiness.status == "ready",
        openai_issues=openai_readiness.issues,
    )
    if not gate.allowed:
        st.warning("Answer blocked by the evidence/readiness gate:")
        for reason in gate.reasons:
            st.write(f"- {reason}")
        return

    answer = synthesize_answer_from_query_payload(payload, question=question, dry_run=False)
    st.write(answer.answer_text)
    st.caption(f"status: {answer.status} | model: {answer.model} | sources: {answer.source_count}")
    st.write("Validated citations")
    st.dataframe(answer_citation_rows(answer.to_dict()), use_container_width=True)
    if answer.rejected_citations:
        st.error(f"Rejected (hallucinated) citations were dropped: {answer.rejected_citations}")


if __name__ == "__main__":
    main()
