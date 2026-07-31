"""Unified analyst brief view: one screen pairing cited filing evidence with
options-market context, with an opt-in gated answer.

This is a separate evidence-first view; it does not modify or merge into the
volatility dashboard. Market context is attached through the existing injectable
provider seam and is offline-safe by default.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.financial_rag.api import build_local_api_service
from src.financial_rag.integration import (
    build_unified_brief,
    market_provider_from_metrics,
    volatility_market_provider,
)
from src.dashboard.theme import inject_theme
from src.financial_rag.settings import project_root
from src.financial_rag.workbench import answer_citation_rows, company_options, evidence_rows

DETERMINISTIC_SNAPSHOT = {
    "source_mode": "Fallback",
    "message": "Deterministic offline market snapshot (not live).",
    "front_expected_move_pct": 8.2,
    "iv_rank": 64.0,
    "iv_30d": 0.52,
    "skew": -0.04,
}


def main() -> None:
    st.set_page_config(page_title="Filing + Market Brief", layout="wide")
    inject_theme(st)
    st.title("Filing + Market Brief")
    st.caption(
        "Cited SEC filing evidence and options-market context, side by side and "
        "clearly labeled as different data sources. The answer is opt-in and "
        "OpenAI-only; the volatility dashboard is unchanged."
    )

    with st.sidebar:
        st.header("Query")
        use_voyage = st.checkbox("Use Voyage query embeddings", value=True)
        top_k = st.number_input("Top-k", min_value=1, max_value=20, value=5, step=1)
        per_subquery_k = st.number_input("Per-subquery k", min_value=1, max_value=20, value=8, step=1)
        live_market = st.checkbox("Use live volatility engine for market context", value=False)

    service = build_local_api_service(root=project_root(), use_voyage=use_voyage)
    options = company_options(service.companies()) or ["NVDA"]
    default_index = options.index("NVDA") if "NVDA" in options else 0
    with st.sidebar:
        ticker = st.selectbox("Ticker", options, index=default_index)

    provider: Callable[[str], dict[str, Any]] = (
        volatility_market_provider if live_market else market_provider_from_metrics(DETERMINISTIC_SNAPSHOT)
    )
    question = st.text_area(
        "Question",
        value="How have NVIDIA data center demand disclosures changed over the last year?",
        height=90,
    )

    run_answer = st.checkbox("Generate grounded answer (opt-in, OpenAI)", value=False)
    if not st.button("Build Brief", type="primary"):
        return

    brief = build_unified_brief(
        service,
        question=question,
        ticker=ticker,
        top_k=int(top_k),
        per_subquery_k=int(per_subquery_k),
        market_provider=provider,
        run_answer=run_answer,
    ).to_dict()

    _render_answer(brief)

    evidence_column, market_column = st.columns(2)
    with evidence_column:
        st.subheader("Filing evidence (cited disclosure)")
        st.caption(brief["filing_evidence"]["description"])
        st.dataframe(evidence_rows({"results": _evidence_results(brief)}), use_container_width=True)
        st.write("Validated citations")
        st.json(brief["filing_evidence"]["accepted_citations"])
        if brief["filing_evidence"]["rejected_citations"]:
            st.error(f"Rejected citations: {brief['filing_evidence']['rejected_citations']}")

    with market_column:
        st.subheader("Market context (options-market-implied)")
        market = brief["market_context"]
        st.caption(market["description"])
        st.metric("Status", market["status"])
        st.metric("Source mode", market["source_mode"] or "unavailable")
        st.json(market["metrics"])

    st.subheader("Data sources")
    st.dataframe(brief["data_sources"], use_container_width=True)
    for note in brief["notes"]:
        st.write(f"- {note}")


def _render_answer(brief: dict[str, Any]) -> None:
    st.subheader("Answer (opt-in, grounded in cited evidence)")
    answer = brief["answer"]
    if answer is None:
        gate = brief["answer_gate"]
        st.info(f"No answer generated (gate allowed = {gate['allowed']}).")
        for reason in gate["reasons"]:
            st.write(f"- {reason}")
        return
    st.write(answer["answer_text"])
    st.caption(f"status: {answer.get('status', '')} | model: {answer.get('model', '')}")
    st.write("Validated citations")
    st.dataframe(answer_citation_rows(answer), use_container_width=True)
    if answer.get("rejected_citations"):
        st.error(f"Rejected (hallucinated) citations were dropped: {answer['rejected_citations']}")


def _evidence_results(brief: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item.get("label", ""),
            "citation_label": item.get("label", ""),
            "source_url": item.get("source_url", ""),
            "source_excerpt": item.get("excerpt", ""),
            "metadata": {
                "ticker": item.get("ticker", ""),
                "form_type": item.get("form_type", ""),
                "filing_date": item.get("filing_date", ""),
            },
        }
        for item in brief["filing_evidence"]["evidence"]
    ]


if __name__ == "__main__":
    main()
