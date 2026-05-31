"""Gold-label resolution for local filings retrieval evals.

The label specs are intentionally stored as human-reviewable selectors rather
than checked-in generated chunk IDs. Running the repair/eval scripts resolves
the selectors to the current local chunk IDs after reingestion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from src.financial_rag.evaluation.expanded import EXPANDED_RETRIEVAL_CASES, ExpandedEvalCase
from src.financial_rag.retrieval import LocalChunkRecord, lexical_relevance_score


@dataclass(frozen=True)
class GoldLabelSpec:
    case_id: str
    topic: str
    ticker: str
    required_terms: tuple[str, ...]
    form_types: tuple[str, ...] = ()
    document_roles: tuple[str, ...] = ()
    exhibit_types: tuple[str, ...] = ()
    item_numbers: tuple[str, ...] = ()
    max_labels: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedGoldLabel:
    case_id: str
    topic: str
    ticker: str
    chunk_id: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GOLD_LABEL_SPECS: tuple[GoldLabelSpec, ...] = (
    GoldLabelSpec("nvda-item1a-export-controls", "NVDA Item 1A export controls", "NVDA", ("export", "control"), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("nvda-item1a-supply", "NVDA Item 1A supply", "NVDA", ("supply",), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("nvda-10q-risk-update", "NVDA 10-Q risk update", "NVDA", ("risk",), ("10-Q",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("nvda-cfo-revenue", "NVDA CFO revenue commentary", "NVDA", ("revenue",), ("EX-99",), ("exhibit",), ("CFO_COMMENTARY",), max_labels=1),
    GoldLabelSpec("nvda-press-release-gross-margin", "NVDA press release gross margin", "NVDA", ("gross", "margin"), ("EX-99",), ("exhibit",), ("PRESS_RELEASE",), max_labels=1),
    GoldLabelSpec("nvda-data-center-demand", "NVDA data center demand", "NVDA", ("data center",), max_labels=2),
    GoldLabelSpec("nvda-inventory-capacity", "NVDA inventory capacity", "NVDA", ("inventory", "capacity"), max_labels=1),
    GoldLabelSpec("nvda-safe-harbor-negative-control", "NVDA concrete risks", "NVDA", ("risk",), ("10-K",), item_numbers=("1A",), max_labels=5),
    GoldLabelSpec("amd-item1a-supply", "AMD Item 1A supply", "AMD", ("supply",), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("amd-data-center", "AMD data center", "AMD", ("data center", "revenue"), max_labels=2),
    GoldLabelSpec("amd-export-controls", "AMD export controls", "AMD", ("export",), ("10-K",), max_labels=1),
    GoldLabelSpec("msft-item1a-ai-risk", "MSFT Item 1A AI cloud risk", "MSFT", ("risk",), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("msft-capital-allocation", "MSFT capital allocation", "MSFT", ("repurchase",), max_labels=1),
    GoldLabelSpec("msft-azure-revenue", "MSFT Azure revenue", "MSFT", ("azure", "revenue"), max_labels=2),
    GoldLabelSpec("aapl-item1a-supply", "AAPL Item 1A supply", "AAPL", ("supply",), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("aapl-china-risk", "AAPL China risk", "AAPL", ("china",), ("10-K",), max_labels=1),
    GoldLabelSpec("aapl-gross-margin", "AAPL gross margin", "AAPL", ("gross margin",), max_labels=1),
    GoldLabelSpec("jpm-item1a-credit-risk", "JPM Item 1A credit risk", "JPM", ("credit", "risk"), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("jpm-capital-return", "JPM capital return", "JPM", ("capital",), max_labels=1),
    GoldLabelSpec("jpm-interest-rate-risk", "JPM interest rate risk", "JPM", ("interest rate", "risk"), max_labels=1),
    GoldLabelSpec("xom-item1a-commodity-risk", "XOM Item 1A commodity risk", "XOM", ("commodity", "price"), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("xom-capital-spending", "XOM capital spending", "XOM", ("capital", "spending"), max_labels=2),
    GoldLabelSpec("xom-energy-transition", "XOM energy transition", "XOM", ("energy transition",), max_labels=1),
    GoldLabelSpec("nvda-amd-data-center-compare", "NVDA data center comparison", "NVDA", ("data center",), max_labels=1),
    GoldLabelSpec("nvda-amd-data-center-compare", "AMD data center comparison", "AMD", ("data center",), max_labels=1),
    GoldLabelSpec("msft-item1a-cybersecurity", "MSFT Item 1A cybersecurity risk", "MSFT", ("cybersecurity",), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("aapl-services", "AAPL services revenue", "AAPL", ("services", "revenue"), max_labels=2),
    GoldLabelSpec("aapl-press-release-revenue", "AAPL press release revenue", "AAPL", ("revenue",), ("EX-99",), ("exhibit",), ("PRESS_RELEASE",), max_labels=1),
    GoldLabelSpec("aapl-item1a-competition", "AAPL Item 1A competition risk", "AAPL", ("competition",), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("jpm-consumer-banking", "JPM consumer and community banking", "JPM", ("consumer",), max_labels=2),
    GoldLabelSpec("jpm-net-interest-income", "JPM net interest income", "JPM", ("net interest income",), max_labels=2),
    GoldLabelSpec("xom-press-release-earnings", "XOM press release earnings", "XOM", ("earnings",), ("EX-99",), ("exhibit",), ("PRESS_RELEASE",), max_labels=2),
    GoldLabelSpec("googl-cloud-revenue", "GOOGL Google Cloud revenue", "GOOGL", ("cloud", "revenue"), max_labels=2),
    GoldLabelSpec("googl-item1a-competition", "GOOGL Item 1A competition risk", "GOOGL", ("competition",), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("meta-advertising-revenue", "META advertising revenue", "META", ("advertising", "revenue"), max_labels=2),
    GoldLabelSpec("meta-reality-labs", "META Reality Labs", "META", ("reality labs",), max_labels=1),
    GoldLabelSpec("meta-item1a-regulation", "META Item 1A regulation risk", "META", ("regulation",), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("amzn-aws-revenue", "AMZN AWS revenue", "AMZN", ("aws",), max_labels=2),
    GoldLabelSpec("bac-item1a-credit-risk", "BAC Item 1A credit risk", "BAC", ("credit", "risk"), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("bac-consumer-banking", "BAC consumer banking", "BAC", ("consumer",), max_labels=2),
    GoldLabelSpec("gs-item1a-market-risk", "GS Item 1A market risk", "GS", ("market", "risk"), ("10-K",), item_numbers=("1A",), max_labels=1),
    GoldLabelSpec("gs-trading", "GS trading and global markets", "GS", ("trading",), max_labels=2),
    # INTC selectors avoid item-number filters: Intel's filing labels live only in
    # a trailing cross-reference index, so its body chunks carry no item metadata.
    GoldLabelSpec("intc-data-center", "INTC data center", "INTC", ("data center",), max_labels=2),
    GoldLabelSpec("intc-manufacturing", "INTC manufacturing and foundry", "INTC", ("manufacturing",), max_labels=2),
    GoldLabelSpec("intc-competition", "INTC competition", "INTC", ("competition",), max_labels=1),
)


def resolve_gold_labels(
    chunks: Sequence[LocalChunkRecord],
    *,
    specs: Sequence[GoldLabelSpec] = GOLD_LABEL_SPECS,
) -> list[ResolvedGoldLabel]:
    """Resolve human-selected label specs to current local chunk IDs."""

    labels: list[ResolvedGoldLabel] = []
    questions_by_case = {case.case_id: case.question for case in EXPANDED_RETRIEVAL_CASES}
    for spec in specs:
        question = questions_by_case.get(spec.case_id, spec.topic)
        candidates = [
            (score, chunk)
            for chunk in chunks
            if (score := _candidate_score(chunk, spec, question=question)) > 0
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)
        for score, chunk in candidates[: spec.max_labels]:
            labels.append(
                ResolvedGoldLabel(
                    case_id=spec.case_id,
                    topic=spec.topic,
                    ticker=spec.ticker,
                    chunk_id=chunk.chunk_id,
                    score=score,
                    metadata={
                        key: chunk.metadata.get(key, "")
                        for key in (
                            "ticker",
                            "form_type",
                            "filing_date",
                            "accession_number",
                            "document_role",
                            "exhibit_type",
                            "item_number",
                            "section_path",
                        )
                    },
                )
            )
    return labels


def apply_gold_labels_to_cases(
    cases: Sequence[ExpandedEvalCase],
    labels: Sequence[ResolvedGoldLabel],
) -> tuple[ExpandedEvalCase, ...]:
    """Return eval cases with resolved relevant chunk IDs attached."""

    labels_by_case: dict[str, set[str]] = {}
    for label in labels:
        labels_by_case.setdefault(label.case_id, set()).add(label.chunk_id)

    labeled_cases: list[ExpandedEvalCase] = []
    for case in cases:
        chunk_ids = labels_by_case.get(case.case_id)
        if chunk_ids is None:
            labeled_cases.append(case)
            continue
        labeled_cases.append(
            ExpandedEvalCase(
                case.case_id,
                case.question,
                case.tickers,
                case.expected_query_type,
                case.source_constraints,
                relevant_chunk_ids=frozenset(chunk_ids),
                answer_eval=case.answer_eval,
                notes=case.notes,
            )
        )
    return tuple(labeled_cases)


def gold_label_summary(labels: Sequence[ResolvedGoldLabel]) -> dict[str, Any]:
    return {
        "label_count": len(labels),
        "case_count": len({label.case_id for label in labels}),
        "companies": sorted({label.ticker for label in labels}),
        "by_company": {
            ticker: sum(1 for label in labels if label.ticker == ticker)
            for ticker in sorted({label.ticker for label in labels})
        },
    }


def _candidate_score(chunk: LocalChunkRecord, spec: GoldLabelSpec, *, question: str) -> float:
    metadata = chunk.metadata
    if _norm(metadata.get("ticker")) != spec.ticker:
        return 0.0
    if spec.form_types and _norm(metadata.get("form_type")) not in {_norm(value) for value in spec.form_types}:
        return 0.0
    if spec.document_roles and _norm(metadata.get("document_role")) not in {_norm(value) for value in spec.document_roles}:
        return 0.0
    if spec.exhibit_types and _norm(metadata.get("exhibit_type")) not in {_norm(value) for value in spec.exhibit_types}:
        return 0.0
    if spec.item_numbers and _norm(metadata.get("item_number")) not in {_norm(value) for value in spec.item_numbers}:
        return 0.0

    text = chunk.chunk_text.lower()
    matched = sum(1 for term in spec.required_terms if term.lower() in text)
    if matched != len(spec.required_terms):
        return 0.0
    metadata_score = 0.0
    if spec.item_numbers:
        metadata_score += 2.0
    if spec.exhibit_types:
        metadata_score += 1.0
    return (
        float(matched)
        + metadata_score
        + (lexical_relevance_score(question, chunk.chunk_text, chunk.metadata) * 5)
        + min(len(text), 2200) / 4400
    )


def _norm(value: object) -> str:
    return str(value).strip().upper()


def expanded_cases_with_current_gold(chunks: Sequence[LocalChunkRecord]) -> tuple[ExpandedEvalCase, ...]:
    return apply_gold_labels_to_cases(EXPANDED_RETRIEVAL_CASES, resolve_gold_labels(chunks))
