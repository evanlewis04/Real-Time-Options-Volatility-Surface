"""Phase 1 Stage 1 — point-in-time backfill logic (offline, pure)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "financial_rag_backfill_pit",
    Path(__file__).resolve().parents[2] / "scripts" / "financial_rag_backfill_pit.py",
)
backfill = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(backfill)


def test_period_end_derives_from_nested_report_date() -> None:
    assert backfill.period_end_for_row({"metadata": {"report_date": "2025-09-27"}}) == "2025-09-27"
    assert backfill.period_end_for_row({"report_date": "2025-06-30"}) == "2025-06-30"
    # Existing period_end wins and is not overwritten.
    assert backfill.period_end_for_row({"period_end": "2024-01-01", "report_date": "2025-06-30"}) == "2024-01-01"
    assert backfill.period_end_for_row({}) == ""


def test_acceptance_map_from_payloads_normalizes_and_skips_missing() -> None:
    payload = {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K"],
                "accessionNumber": ["0000034088-26-000045", "0001045810-26-000051"],
                "filingDate": ["2026-02-18", "2026-05-20"],
                "reportDate": ["2025-12-31", "2026-05-20"],
                "primaryDocument": ["a.htm", "b.htm"],
                "primaryDocDescription": ["10-K", "8-K"],
                "acceptanceDateTime": ["2026-02-18T17:02:05.000Z", ""],
            }
        }
    }

    mapping = backfill.acceptance_map_from_payloads([payload])

    assert mapping == {"0000034088-26-000045": "2026-02-18T17:02:05+00:00"}


def test_backfill_row_fills_missing_and_is_idempotent() -> None:
    acceptance = {"0000034088-26-000045": "2026-02-18T17:02:05+00:00"}
    row = {
        "accession_number": "0000034088-26-000045",
        "metadata": {"report_date": "2025-12-31"},
    }

    assert backfill.backfill_row(row, acceptance) is True
    assert row["filed_at"] == "2026-02-18T17:02:05+00:00"
    assert row["period_end"] == "2025-12-31"

    # Second pass changes nothing (idempotent).
    assert backfill.backfill_row(row, acceptance) is False


def test_backfill_row_flags_unresolved_acceptance_without_faking() -> None:
    row = {"accession_number": "9999999999-99-999999", "metadata": {"report_date": "2025-12-31"}}

    changed = backfill.backfill_row(row, {})

    assert changed is True
    assert row["period_end"] == "2025-12-31"
    assert row["filed_at"] == ""  # flagged empty, never faked
