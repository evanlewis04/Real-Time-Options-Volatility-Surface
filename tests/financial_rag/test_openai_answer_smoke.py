from src.financial_rag.synthesis import (
    DEFAULT_OPENAI_SYNTHESIS_MODEL,
    check_openai_readiness,
    synthesize_answer_from_query_payload,
    validate_answer_citations,
)


def test_openai_readiness_reports_key_and_sdk_state_without_calling_api(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")

    readiness = check_openai_readiness(model=DEFAULT_OPENAI_SYNTHESIS_MODEL)

    assert readiness.model == DEFAULT_OPENAI_SYNTHESIS_MODEL
    assert readiness.api_key_configured is False
    assert readiness.status in {"ready", "not_ready"}
    assert readiness.issues


def test_dry_run_answer_builds_prompt_and_validates_citations() -> None:
    payload = _query_payload()

    answer = synthesize_answer_from_query_payload(
        payload,
        question="What risks does NVIDIA describe?",
        dry_run=True,
    )

    assert answer.status == "pass"
    assert answer.dry_run is True
    assert answer.accepted_citations[0]["label"] == "S1"
    assert answer.rejected_citations == []
    assert "Every factual sentence" in answer.prompt_preview
    assert "Risk factors include export controls" in answer.prompt_preview


def test_live_answer_path_uses_injected_client_and_rejects_hallucinated_citations() -> None:
    payload = _query_payload()

    answer = synthesize_answer_from_query_payload(
        payload,
        question="What risks does NVIDIA describe?",
        dry_run=False,
        client=_FakeResponsesClient("NVIDIA cites export control risks [S1]. This invented point is unsupported [S9]."),
    )

    assert answer.status == "warning"
    assert [citation["label"] for citation in answer.accepted_citations] == ["S1"]
    assert answer.rejected_citations == ["S9"]


def test_validate_answer_citations_accepts_only_retrieved_labels() -> None:
    accepted, rejected = validate_answer_citations("Known risk [S1]. Unknown [S8].", _query_payload())

    assert accepted[0]["chunk_id"] == "risk"
    assert rejected == ["S8"]


class _FakeResponsesClient:
    def __init__(self, text: str) -> None:
        self.text = text

    def create_response(self, *, model: str, instructions: str, input_text: str) -> str:
        assert model
        assert "provided evidence" in instructions
        assert "[S1]" in input_text
        return self.text


def _query_payload() -> dict[str, object]:
    return {
        "results": [
            {
                "chunk_id": "risk",
                "citation_label": "S1",
                "source_url": "https://www.sec.gov/Archives/risk.htm",
                "source_excerpt": "Risk factors include export controls and demand uncertainty.",
                "metadata": {
                    "ticker": "NVDA",
                    "form_type": "10-K",
                    "filing_date": "2026-02-01",
                    "accession_number": "risk-accession",
                },
            },
            {
                "chunk_id": "release",
                "citation_label": "S2",
                "source_url": "https://www.sec.gov/Archives/release.htm",
                "source_excerpt": "The company described strong demand.",
                "metadata": {
                    "ticker": "NVDA",
                    "form_type": "EX-99",
                    "filing_date": "2026-02-01",
                    "accession_number": "release-accession",
                },
            },
        ]
    }
