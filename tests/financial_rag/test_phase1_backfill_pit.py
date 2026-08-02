"""Phase 1 Stage 1 — point-in-time backfill logic (offline, pure)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.financial_rag.storage import LocalRagStore

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


def test_run_backfill_dry_run_previews_without_writing(tmp_path: Path) -> None:
    store = LocalRagStore(root=tmp_path)
    chunk_path = store.chunks_dir / "DOC.jsonl"
    row = {
        "chunk_id": "c1",
        "accession_number": "0000000000-26-000001",
        "report_date": "2025-12-31",
    }
    chunk_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    before = chunk_path.read_text(encoding="utf-8")

    preview = backfill.run_backfill(store, {}, dry_run=True)

    assert preview["dry_run"] is True and preview["chunks_changed"] == 1
    assert chunk_path.read_text(encoding="utf-8") == before  # no write on a dry run

    applied = backfill.run_backfill(store, {}, dry_run=False)

    assert applied["chunks_changed"] == 1
    written = json.loads(chunk_path.read_text(encoding="utf-8").splitlines()[0])
    assert written["period_end"] == "2025-12-31"  # execute path actually writes


def test_execute_path_loads_env_to_resolve_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: main(--execute) must load .env so SEC_USER_AGENT resolves.

    The original backfill omitted ``load_environment()``, so the acceptance re-fetch
    silently degraded to offline-only (period_end filled, filed_at flagged empty).
    Simulate the secret living only in .env — ``load_environment()`` is the sole thing
    that sets it — and assert the network client gets built. If main() failed to load
    the env (the bug), the secret would stay unset and SECClient would never construct.
    """
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)

    def fake_load_env() -> None:
        monkeypatch.setenv("SEC_USER_AGENT", "Regression Agent ops@corp.test")

    monkeypatch.setattr(backfill, "load_environment", fake_load_env)

    constructed: dict[str, str] = {}

    class FakeSECClient:
        def __init__(self, user_agent: str) -> None:
            constructed["user_agent"] = user_agent

        def fetch_company_submissions(self, cik: str) -> dict:
            return {"filings": {"recent": {}}}

    monkeypatch.setattr(backfill, "SECClient", FakeSECClient)

    store = LocalRagStore(root=tmp_path)
    (store.chunks_dir / "DOC.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "c1",
                "cik": "0000034088",
                "accession_number": "0000034088-26-000045",
                "report_date": "2025-12-31",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["financial_rag_backfill_pit", "--execute", "--root", str(tmp_path)])

    assert backfill.main() == 0
    assert constructed.get("user_agent") == "Regression Agent ops@corp.test"
