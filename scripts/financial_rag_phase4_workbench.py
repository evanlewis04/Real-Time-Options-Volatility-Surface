"""Streamlit filings intelligence workbench for local Phase 4 evidence review."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.financial_rag.api import LocalApiError, QueryRequest, build_local_api_service
from src.financial_rag.audit import build_evidence_quality_report, build_readiness_report
from src.financial_rag.settings import project_root
from src.financial_rag.workbench import (
    change_rows,
    coverage_rows,
    evidence_quality_issue_rows,
    evidence_rows,
    language_signal_rows,
    readiness_issue_rows,
    rejected_citation_rows,
)


def main() -> None:
    st.set_page_config(page_title="Filings Intelligence Workbench", layout="wide")
    st.title("Filings Intelligence Workbench")
    st.caption("Local SEC filings evidence review. No answer synthesis or SEC refetch.")

    with st.sidebar:
        ticker = st.text_input("Ticker", value="NVDA").strip().upper() or "NVDA"
        top_k = st.number_input("Top-k", min_value=1, max_value=20, value=5, step=1)
        per_subquery_k = st.number_input("Per-subquery k", min_value=1, max_value=20, value=5, step=1)
        use_voyage = st.checkbox("Use Voyage query embeddings", value=True)

    question = st.text_area(
        "Question",
        value="How have NVIDIA risk disclosures changed over the last year?",
        height=90,
    )
    service = build_local_api_service(root=project_root(), use_voyage=use_voyage)
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

    if st.button("Retrieve Evidence", type="primary"):
        try:
            payload = service.query(
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
            return
        routed = payload["routed_query"]
        st.subheader("Routing")
        st.json(routed)

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


if __name__ == "__main__":
    main()
