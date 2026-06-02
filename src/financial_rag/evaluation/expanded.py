"""Expanded retrieval and answer eval scaffolding for local filings RAG."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.financial_rag.audit import build_evidence_quality_report
from src.financial_rag.synthesis import EvidenceAnswer, synthesize_answer_from_query_payload


SAFE_HARBOR_TERMS = ("safe harbor", "forward-looking", "undue reliance", "actual results may differ")

# Tickers with a locally cached SEC corpus. Cases outside this set are treated as
# unsupported-ticker controls. Keep this in sync with the local cache.
CACHED_TICKERS: tuple[str, ...] = (
    "NVDA",
    "AMD",
    "MSFT",
    "AAPL",
    "JPM",
    "XOM",
    "INTC",
    "GOOGL",
    "META",
    "AMZN",
    "BAC",
    "GS",
)


@dataclass(frozen=True)
class SourceConstraint:
    ticker: str
    form_type: str | None = None
    document_role: str | None = None
    exhibit_type: str | None = None
    item_number: str | None = None
    accession: str | None = None
    filing_date: str | None = None
    required_keywords: tuple[str, ...] = ()
    forbidden_keywords: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpandedEvalCase:
    case_id: str
    question: str
    tickers: tuple[str, ...]
    expected_query_type: str
    source_constraints: tuple[SourceConstraint, ...]
    relevant_chunk_ids: frozenset[str] = frozenset()
    answer_eval: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relevant_chunk_ids"] = sorted(self.relevant_chunk_ids)
        payload["source_constraints"] = [constraint.to_dict() for constraint in self.source_constraints]
        return payload


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    status: str
    retrieved_count: int
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    source_hit: bool
    metadata_complete: bool
    evidence_quality_status: str
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerCaseResult:
    case_id: str
    status: str
    dry_run: bool
    accepted_citation_count: int
    rejected_citation_count: int
    uncited_sentence_count: int
    weak_evidence: bool
    insufficient_evidence: bool
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EXPANDED_RETRIEVAL_CASES: tuple[ExpandedEvalCase, ...] = (
    ExpandedEvalCase(
        "nvda-item1a-export-controls",
        "What does NVIDIA Item 1A say about export controls?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA", form_type="10-K", item_number="1A", required_keywords=("export", "control")),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "nvda-item1a-supply",
        "What risks does NVIDIA describe around supply constraints?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA", form_type="10-K", item_number="1A", required_keywords=("supply",)),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "nvda-10q-risk-update",
        "What did NVIDIA update in recent 10-Q risk language?",
        ("NVDA",),
        "temporal",
        (SourceConstraint("NVDA", form_type="10-Q", required_keywords=("risk",)),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "nvda-cfo-revenue",
        "What did NVIDIA CFO commentary say about revenue?",
        ("NVDA",),
        "speaker_specific",
        (SourceConstraint("NVDA", form_type="EX-99", document_role="exhibit", exhibit_type="CFO_COMMENTARY", required_keywords=("revenue",)),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "nvda-press-release-gross-margin",
        "What does NVIDIA's press release say about gross margin?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA", form_type="EX-99", exhibit_type="PRESS_RELEASE", required_keywords=("gross", "margin")),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "nvda-data-center-demand",
        "What does NVIDIA say about data center demand?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA", required_keywords=("data center", "demand")),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "nvda-inventory-capacity",
        "How does NVIDIA discuss inventory and capacity commitments?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA", required_keywords=("inventory", "capacity")),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "nvda-safe-harbor-negative-control",
        "What concrete operating risks does NVIDIA describe beyond safe harbor language?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA", forbidden_keywords=("safe harbor", "forward-looking")),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "nvda-capital-allocation",
        "What does NVIDIA say about capital allocation or shareholder returns?",
        ("NVDA",),
        "single_doc_lookup",
        (SourceConstraint("NVDA", required_keywords=("share repurchase", "dividend", "capital")),),
    ),
    ExpandedEvalCase(
        "nvda-quarterly-revenue-change",
        "How did NVIDIA revenue commentary change between recent quarters?",
        ("NVDA",),
        "temporal",
        (SourceConstraint("NVDA", required_keywords=("revenue",)),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "amd-item1a-supply",
        "What supply-chain risks does AMD describe in Item 1A?",
        ("AMD",),
        "single_doc_lookup",
        (SourceConstraint("AMD", form_type="10-K", item_number="1A", required_keywords=("supply",)),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "amd-data-center",
        "What does AMD say about data center revenue?",
        ("AMD",),
        "single_doc_lookup",
        (SourceConstraint("AMD", required_keywords=("data center", "revenue")),),
    ),
    ExpandedEvalCase(
        "amd-export-controls",
        "Does AMD mention export controls or China restrictions?",
        ("AMD",),
        "single_doc_lookup",
        (SourceConstraint("AMD", form_type="10-K", required_keywords=("export", "China")),),
    ),
    ExpandedEvalCase(
        "msft-item1a-ai-risk",
        "What AI or cloud risks does Microsoft describe in Item 1A?",
        ("MSFT",),
        "single_doc_lookup",
        (SourceConstraint("MSFT", form_type="10-K", item_number="1A", required_keywords=("AI", "cloud")),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "msft-capital-allocation",
        "What does Microsoft say about dividends or share repurchases?",
        ("MSFT",),
        "single_doc_lookup",
        (SourceConstraint("MSFT", required_keywords=("dividend", "repurchase")),),
    ),
    ExpandedEvalCase(
        "msft-azure-revenue",
        "What does Microsoft say about Azure or cloud revenue growth?",
        ("MSFT",),
        "single_doc_lookup",
        (SourceConstraint("MSFT", required_keywords=("Azure", "revenue")),),
    ),
    ExpandedEvalCase(
        "aapl-item1a-supply",
        "What supply chain risks does Apple describe in Item 1A?",
        ("AAPL",),
        "single_doc_lookup",
        (SourceConstraint("AAPL", form_type="10-K", item_number="1A", required_keywords=("supply",)),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "aapl-china-risk",
        "What does Apple disclose about China-related risks?",
        ("AAPL",),
        "single_doc_lookup",
        (SourceConstraint("AAPL", form_type="10-K", required_keywords=("China",)),),
    ),
    ExpandedEvalCase(
        "aapl-gross-margin",
        "What does Apple say about gross margin drivers?",
        ("AAPL",),
        "single_doc_lookup",
        (SourceConstraint("AAPL", required_keywords=("gross margin",)),),
    ),
    ExpandedEvalCase(
        "jpm-item1a-credit-risk",
        "What credit risks does JPMorgan describe in Item 1A?",
        ("JPM",),
        "single_doc_lookup",
        (SourceConstraint("JPM", form_type="10-K", item_number="1A", required_keywords=("credit", "risk")),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "jpm-capital-return",
        "What does JPMorgan say about capital return or buybacks?",
        ("JPM",),
        "single_doc_lookup",
        (SourceConstraint("JPM", required_keywords=("capital", "repurchase")),),
    ),
    ExpandedEvalCase(
        "jpm-interest-rate-risk",
        "How does JPMorgan discuss interest rate risk?",
        ("JPM",),
        "single_doc_lookup",
        (SourceConstraint("JPM", required_keywords=("interest rate", "risk")),),
    ),
    ExpandedEvalCase(
        "xom-item1a-commodity-risk",
        "What commodity price risks does Exxon Mobil describe in Item 1A?",
        ("XOM",),
        "single_doc_lookup",
        (SourceConstraint("XOM", form_type="10-K", item_number="1A", required_keywords=("commodity", "price")),),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "xom-capital-spending",
        "What does Exxon Mobil say about capital spending?",
        ("XOM",),
        "single_doc_lookup",
        (SourceConstraint("XOM", required_keywords=("capital", "spending")),),
    ),
    ExpandedEvalCase(
        "xom-energy-transition",
        "What does Exxon Mobil disclose about energy transition risks?",
        ("XOM",),
        "single_doc_lookup",
        (SourceConstraint("XOM", required_keywords=("energy transition", "emissions")),),
    ),
    ExpandedEvalCase(
        "nvda-amd-data-center-compare",
        "Compare NVDA and AMD data center commentary.",
        ("NVDA", "AMD"),
        "cross_company",
        (
            SourceConstraint("NVDA", required_keywords=("data center",)),
            SourceConstraint("AMD", required_keywords=("data center",)),
        ),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "aapl-msft-capital-allocation-compare",
        "Compare Apple and Microsoft capital allocation commentary.",
        ("AAPL", "MSFT"),
        "cross_company",
        (
            SourceConstraint("AAPL", required_keywords=("repurchase",)),
            SourceConstraint("MSFT", required_keywords=("dividend",)),
        ),
    ),
    ExpandedEvalCase(
        "nvda-cross-source-risk-cfo",
        "Does NVIDIA 10-Q risk language match CFO commentary?",
        ("NVDA",),
        "cross_source",
        (
            SourceConstraint("NVDA", form_type="10-Q", required_keywords=("risk",)),
            SourceConstraint("NVDA", form_type="EX-99", exhibit_type="CFO_COMMENTARY"),
        ),
        answer_eval=True,
    ),
    ExpandedEvalCase(
        "nvda-press-release-vs-10q-demand",
        "Compare NVIDIA press release demand commentary with the latest 10-Q.",
        ("NVDA",),
        "cross_source",
        (
            SourceConstraint("NVDA", form_type="EX-99", exhibit_type="PRESS_RELEASE", required_keywords=("demand",)),
            SourceConstraint("NVDA", form_type="10-Q", required_keywords=("demand",)),
        ),
    ),
    ExpandedEvalCase(
        "msft-item1a-cybersecurity",
        "What cybersecurity risks does Microsoft describe in Item 1A?",
        ("MSFT",),
        "single_doc_lookup",
        (SourceConstraint("MSFT", form_type="10-K", item_number="1A", required_keywords=("cybersecurity",)),),
    ),
    ExpandedEvalCase(
        "aapl-services",
        "What does Apple say about Services revenue?",
        ("AAPL",),
        "single_doc_lookup",
        (SourceConstraint("AAPL", required_keywords=("services", "revenue")),),
    ),
    ExpandedEvalCase(
        "aapl-press-release-revenue",
        "What does Apple's press release say about revenue?",
        ("AAPL",),
        "single_doc_lookup",
        (SourceConstraint("AAPL", form_type="EX-99", exhibit_type="PRESS_RELEASE", required_keywords=("revenue",)),),
    ),
    ExpandedEvalCase(
        "aapl-item1a-competition",
        "What competition risks does Apple describe in Item 1A?",
        ("AAPL",),
        "single_doc_lookup",
        (SourceConstraint("AAPL", form_type="10-K", item_number="1A", required_keywords=("competition",)),),
    ),
    ExpandedEvalCase(
        "jpm-consumer-banking",
        "What does JPMorgan say about consumer and community banking?",
        ("JPM",),
        "single_doc_lookup",
        (SourceConstraint("JPM", required_keywords=("consumer",)),),
    ),
    ExpandedEvalCase(
        "jpm-net-interest-income",
        "What does JPMorgan say about net interest income?",
        ("JPM",),
        "single_doc_lookup",
        (SourceConstraint("JPM", required_keywords=("net interest income",)),),
    ),
    ExpandedEvalCase(
        "xom-press-release-earnings",
        "What does Exxon Mobil's press release say about earnings?",
        ("XOM",),
        "single_doc_lookup",
        (SourceConstraint("XOM", form_type="EX-99", exhibit_type="PRESS_RELEASE", required_keywords=("earnings",)),),
    ),
    ExpandedEvalCase(
        "googl-cloud-revenue",
        "What does Alphabet say about Google Cloud revenue?",
        ("GOOGL",),
        "single_doc_lookup",
        (SourceConstraint("GOOGL", required_keywords=("cloud", "revenue")),),
    ),
    ExpandedEvalCase(
        "googl-item1a-competition",
        "What competition risks does Alphabet describe in Item 1A?",
        ("GOOGL",),
        "single_doc_lookup",
        (SourceConstraint("GOOGL", form_type="10-K", item_number="1A", required_keywords=("competition",)),),
    ),
    ExpandedEvalCase(
        "meta-advertising-revenue",
        "What does Meta say about advertising revenue?",
        ("META",),
        "single_doc_lookup",
        (SourceConstraint("META", required_keywords=("advertising", "revenue")),),
    ),
    ExpandedEvalCase(
        "meta-reality-labs",
        "What does Meta say about Reality Labs?",
        ("META",),
        "single_doc_lookup",
        (SourceConstraint("META", required_keywords=("reality labs",)),),
    ),
    ExpandedEvalCase(
        "meta-item1a-regulation",
        "What regulatory risks does Meta describe in Item 1A?",
        ("META",),
        "single_doc_lookup",
        (SourceConstraint("META", form_type="10-K", item_number="1A", required_keywords=("regulation",)),),
    ),
    ExpandedEvalCase(
        "amzn-aws-revenue",
        "What does Amazon say about AWS revenue?",
        ("AMZN",),
        "single_doc_lookup",
        (SourceConstraint("AMZN", required_keywords=("aws",)),),
    ),
    ExpandedEvalCase(
        "bac-item1a-credit-risk",
        "What credit risks does Bank of America describe in Item 1A?",
        ("BAC",),
        "single_doc_lookup",
        (SourceConstraint("BAC", form_type="10-K", item_number="1A", required_keywords=("credit", "risk")),),
    ),
    ExpandedEvalCase(
        "bac-consumer-banking",
        "What does Bank of America say about consumer banking?",
        ("BAC",),
        "single_doc_lookup",
        (SourceConstraint("BAC", required_keywords=("consumer",)),),
    ),
    ExpandedEvalCase(
        "gs-item1a-market-risk",
        "What market risks does Goldman Sachs describe in Item 1A?",
        ("GS",),
        "single_doc_lookup",
        (SourceConstraint("GS", form_type="10-K", item_number="1A", required_keywords=("market", "risk")),),
    ),
    ExpandedEvalCase(
        "gs-trading",
        "What does Goldman Sachs say about trading or global markets revenue?",
        ("GS",),
        "single_doc_lookup",
        (SourceConstraint("GS", required_keywords=("trading",)),),
    ),
    ExpandedEvalCase(
        "intc-data-center",
        "What does Intel say about data center revenue?",
        ("INTC",),
        "single_doc_lookup",
        (SourceConstraint("INTC", required_keywords=("data center",)),),
    ),
    ExpandedEvalCase(
        "intc-manufacturing",
        "What does Intel say about manufacturing or foundry operations?",
        ("INTC",),
        "single_doc_lookup",
        (SourceConstraint("INTC", required_keywords=("manufacturing",)),),
    ),
    ExpandedEvalCase(
        "intc-competition",
        "What does Intel say about competition in its business?",
        ("INTC",),
        "single_doc_lookup",
        (SourceConstraint("INTC", required_keywords=("competition",)),),
    ),
    ExpandedEvalCase(
        "unsupported-tsla-risk",
        "What risks does Tesla describe in Item 1A?",
        ("TSLA",),
        "single_doc_lookup",
        (SourceConstraint("TSLA", form_type="10-K", item_number="1A", required_keywords=("risk",)),),
    ),
)

EXPANDED_ANSWER_CASES: tuple[ExpandedEvalCase, ...] = tuple(case for case in EXPANDED_RETRIEVAL_CASES if case.answer_eval)


def filter_cases(
    cases: Sequence[ExpandedEvalCase],
    *,
    tickers: Iterable[str] | None = None,
    max_cases: int | None = None,
    answer_only: bool = False,
) -> list[ExpandedEvalCase]:
    selected = [case for case in cases if not answer_only or case.answer_eval]
    if tickers:
        wanted = {ticker.strip().upper() for ticker in tickers if ticker.strip()}
        selected = [case for case in selected if wanted & set(case.tickers)]
    if max_cases is not None:
        selected = selected[:max_cases]
    return selected


def build_retrieval_quality_report(
    cases: Sequence[ExpandedEvalCase],
    payloads_by_case_id: dict[str, dict[str, Any]],
    *,
    k: int = 5,
) -> dict[str, Any]:
    results = [evaluate_retrieval_case(case, payloads_by_case_id.get(case.case_id, {}), k=k) for case in cases]
    relevant_case_ids = {case.case_id for case in cases if case.relevant_chunk_ids}
    relevant_results = [result for result in results if result.case_id in relevant_case_ids]
    return {
        "k": k,
        "case_count": len(cases),
        "relevant_labeled_case_count": len(relevant_case_ids),
        "company_count": len({ticker for case in cases for ticker in case.tickers}),
        "companies": sorted({ticker for case in cases for ticker in case.tickers}),
        "mean_recall_at_k": _mean([result.recall_at_k for result in relevant_results]),
        "mrr": _mean([result.reciprocal_rank for result in relevant_results]),
        "mean_ndcg_at_k": _mean([result.ndcg_at_k for result in relevant_results]),
        "section_source_hit_rate": _rate(result.source_hit for result in results),
        "metadata_completeness_rate": _rate(result.metadata_complete for result in results),
        "evidence_quality_pass_rate": _rate(result.evidence_quality_status == "pass" for result in results),
        "failure_counts": _failure_counts(results),
        "results": [result.to_dict() for result in results],
    }


def evaluate_retrieval_case(case: ExpandedEvalCase, query_payload: dict[str, Any], *, k: int = 5) -> RetrievalCaseResult:
    if "error" in query_payload:
        code = str(query_payload["error"].get("code", "query_error"))
        return RetrievalCaseResult(
            case_id=case.case_id,
            status="fail",
            retrieved_count=0,
            recall_at_k=0.0,
            reciprocal_rank=0.0,
            ndcg_at_k=0.0,
            source_hit=False,
            metadata_complete=False,
            evidence_quality_status="fail",
            failures=[code],
        )
    results = list(query_payload.get("results", []))
    retrieved_ids = [str(result.get("chunk_id", "")) for result in results]
    source_hit = _source_hit(case.source_constraints, results)
    metadata_complete = _metadata_complete(results)
    evidence_quality = build_evidence_quality_report(query_payload)
    failures = classify_retrieval_failures(case, query_payload, source_hit=source_hit, metadata_complete=metadata_complete)
    recall = _recall_at_k(retrieved_ids, set(case.relevant_chunk_ids), k=k)
    rr = _reciprocal_rank(retrieved_ids, set(case.relevant_chunk_ids))
    ndcg = _ndcg_at_k(retrieved_ids, set(case.relevant_chunk_ids), k=k)
    status = "pass" if source_hit and metadata_complete and evidence_quality.status == "pass" else "fail"
    return RetrievalCaseResult(
        case_id=case.case_id,
        status=status,
        retrieved_count=len(results),
        recall_at_k=recall,
        reciprocal_rank=rr,
        ndcg_at_k=ndcg,
        source_hit=source_hit,
        metadata_complete=metadata_complete,
        evidence_quality_status=evidence_quality.status,
        failures=failures,
    )


def classify_retrieval_failures(
    case: ExpandedEvalCase,
    query_payload: dict[str, Any],
    *,
    source_hit: bool,
    metadata_complete: bool,
) -> list[str]:
    results = list(query_payload.get("results", []))
    failures: list[str] = []
    if not results:
        failures.append("empty_results")
        return failures
    if not source_hit:
        failures.append("wrong_section_or_source")
    if not metadata_complete:
        failures.append("missing_metadata")
    chunk_ids = [str(result.get("chunk_id", "")) for result in results if result.get("chunk_id")]
    if len(chunk_ids) != len(set(chunk_ids)):
        failures.append("duplicate_chunks")
    if _safe_harbor_only(results):
        failures.append("safe_harbor_only")
    if any(ticker not in CACHED_TICKERS for ticker in case.tickers):
        failures.append("unsupported_ticker")
    return sorted(set(failures))


def build_answer_quality_report(
    cases: Sequence[ExpandedEvalCase],
    answers_by_case_id: dict[str, EvidenceAnswer | dict[str, Any]],
) -> dict[str, Any]:
    results = [evaluate_answer_case(case, answers_by_case_id.get(case.case_id)) for case in cases]
    return {
        "case_count": len(cases),
        "company_count": len({ticker for case in cases for ticker in case.tickers}),
        "companies": sorted({ticker for case in cases for ticker in case.tickers}),
        "pass_rate": _rate(result.status == "pass" for result in results),
        "hallucinated_citation_count": sum(result.rejected_citation_count for result in results),
        "uncited_sentence_count": sum(result.uncited_sentence_count for result in results),
        "weak_evidence_count": sum(1 for result in results if result.weak_evidence),
        "insufficient_evidence_count": sum(1 for result in results if result.insufficient_evidence),
        "failure_counts": _answer_failure_counts(results),
        "results": [result.to_dict() for result in results],
    }


def evaluate_answer_case(case: ExpandedEvalCase, answer: EvidenceAnswer | dict[str, Any] | None) -> AnswerCaseResult:
    if answer is None:
        return AnswerCaseResult(case.case_id, "fail", True, 0, 0, 0, False, False, ["missing_answer"])
    payload = answer.to_dict() if isinstance(answer, EvidenceAnswer) else answer
    answer_text = str(payload.get("answer_text", ""))
    rejected = list(payload.get("rejected_citations", []))
    accepted = list(payload.get("accepted_citations", []))
    uncited = count_uncited_factual_sentences(answer_text)
    weak = _text_is_safe_harbor_only(answer_text)
    insufficient = "insufficient" in answer_text.lower() or "missing" in answer_text.lower()
    failures: list[str] = []
    if rejected:
        failures.append("hallucinated_citation")
    if uncited:
        failures.append("uncited_factual_sentence")
    if weak:
        failures.append("weak_or_safe_harbor_only")
    if not accepted:
        failures.append("no_valid_citations")
    status = "pass" if not failures else "warning" if accepted else "fail"
    return AnswerCaseResult(
        case_id=case.case_id,
        status=status,
        dry_run=bool(payload.get("dry_run", True)),
        accepted_citation_count=len(accepted),
        rejected_citation_count=len(rejected),
        uncited_sentence_count=uncited,
        weak_evidence=weak,
        insufficient_evidence=insufficient,
        failures=failures,
    )


def answer_case_from_query_payload(
    case: ExpandedEvalCase,
    query_payload: dict[str, Any],
    *,
    dry_run: bool,
    model: str,
) -> EvidenceAnswer:
    return synthesize_answer_from_query_payload(
        query_payload,
        question=case.question,
        model=model,
        dry_run=dry_run,
    )


def count_uncited_factual_sentences(answer_text: str) -> int:
    count = 0
    for sentence in re.split(r"(?<=[.!?])\s+", answer_text.strip()):
        compact = sentence.strip()
        if len(compact) < 30:
            continue
        if compact.lower().startswith(("dry run only", "no retrieved evidence")):
            continue
        if not re.search(r"\[S\d+\]", compact, flags=re.I):
            count += 1
    return count


def write_json_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_csv_rows(rows: Sequence[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _source_hit(constraints: Sequence[SourceConstraint], results: Sequence[dict[str, Any]]) -> bool:
    if not constraints:
        return True
    return all(any(_matches_constraint(result, constraint) for result in results) for constraint in constraints)


def _matches_constraint(result: dict[str, Any], constraint: SourceConstraint) -> bool:
    metadata = result.get("metadata", {})
    text = _constraint_text(result)
    checks = {
        "ticker": constraint.ticker,
        "form_type": constraint.form_type,
        "document_role": constraint.document_role,
        "exhibit_type": constraint.exhibit_type,
        "item_number": constraint.item_number,
        "accession_number": constraint.accession,
        "filing_date": constraint.filing_date,
    }
    for key, expected in checks.items():
        if expected is None:
            continue
        if str(metadata.get(key, "")).strip().upper() != str(expected).strip().upper():
            return False
    if constraint.required_keywords and not all(
        _keyword_matches_text(keyword, text) for keyword in constraint.required_keywords
    ):
        return False
    if constraint.forbidden_keywords and any(_keyword_matches_text(keyword, text) for keyword in constraint.forbidden_keywords):
        return False
    return True


def _constraint_text(result: dict[str, Any]) -> str:
    parent_context = result.get("parent_context")
    context_text = ""
    if isinstance(parent_context, dict):
        context_text = str(parent_context.get("context_text", ""))
    return f"{result.get('source_excerpt', '')} {context_text} {result.get('source_url', '')}".lower()


def _keyword_matches_text(keyword: str, text: str) -> bool:
    normalized = keyword.lower()
    if normalized in text:
        return True
    return any(alias in text for alias in _keyword_aliases(normalized))


def _keyword_aliases(keyword: str) -> tuple[str, ...]:
    aliases = {
        "ai": ("artificial intelligence", "copilot"),
        "azure": ("cloud services", "intelligent cloud"),
        "china": ("greater china", "chinese"),
        "emissions": ("emission", "lower-emission", "greenhouse gas"),
        "energy transition": ("alternative energy", "lower-emission", "transition"),
        "repurchase": ("repurchased", "repurchases", "buyback", "buybacks", "issuer purchases"),
        "share repurchase": ("share repurchases", "shares repurchased", "issuer purchases", "buybacks"),
        "spending": ("expenditure", "expenditures", "capex", "cash capex"),
        "supply": ("supplier", "suppliers", "supply chain"),
    }
    return aliases.get(keyword, ())


def _metadata_complete(results: Sequence[dict[str, Any]]) -> bool:
    required = ("ticker", "form_type", "filing_date", "accession_number")
    return all(
        result.get("source_url")
        and all(str(result.get("metadata", {}).get(key, "")).strip() for key in required)
        for result in results
    )


def _safe_harbor_only(results: Sequence[dict[str, Any]]) -> bool:
    if not results:
        return False
    return all(any(term in str(result.get("source_excerpt", "")).lower() for term in SAFE_HARBOR_TERMS) for result in results)


def _text_is_safe_harbor_only(text: str) -> bool:
    lower = text.lower()
    return bool(lower.strip()) and any(term in lower for term in SAFE_HARBOR_TERMS) and not any(
        term in lower for term in ("supply", "export", "competition", "credit", "commodity", "margin")
    )


def _recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], *, k: int) -> float:
    if not relevant_ids:
        return 0.0
    return 1.0 if set(retrieved_ids[:k]) & relevant_ids else 0.0


def _reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / index
    return 0.0


def _ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], *, k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _failure_counts(results: Sequence[RetrievalCaseResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for failure in result.failures:
            counts[failure] = counts.get(failure, 0) + 1
    return dict(sorted(counts.items()))


def _answer_failure_counts(results: Sequence[AnswerCaseResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for failure in result.failures:
            counts[failure] = counts.get(failure, 0) + 1
    return dict(sorted(counts.items()))


def _rate(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for item in items if item) / len(items)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
