"""Phase 1 Stage 4 — S&P 500 scaling machinery (mocked SEC client, no live EDGAR).

Covers the three gate items: the constituent CSV parses and resolves to CIKs,
a batched fetch resumes after interruption (skips completed names, re-runs the
interrupted one, no dupes), and an idempotent re-run is a no-op.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.financial_rag.ingestion.batch_fetch import (
    fetch_constituents,
    load_completed_tickers,
)
from src.financial_rag.ingestion.constituents import (
    parse_constituents_csv,
    resolve_constituents,
)
from src.financial_rag.ingestion.sec_client import (
    CompanyRecord,
    recent_filing_records,
)
from src.financial_rag.storage import LocalRagStore

REPO_ROOT = Path(__file__).resolve().parents[2]
CONSTITUENTS_CSV = REPO_ROOT / "config" / "sp500_constituents.csv"


# --------------------------------------------------------------------------- #
# Constituent CSV: parse + resolve
# --------------------------------------------------------------------------- #

def test_parse_constituents_csv_pads_cik_ignores_comments_and_flags_missing() -> None:
    text = (
        "# a provenance comment line\n"
        "# survivorship-bias note\n"
        "ticker,cik,company_name\n"
        "AAPL,320193,Apple Inc.\n"
        "brk.b,1067983,Berkshire Hathaway\n"
        ",999,No Ticker Row\n"
        "NOCIK,,Missing Cik Co\n"
    )

    records = parse_constituents_csv(text)

    by_ticker = {r.ticker: r for r in records}
    assert set(by_ticker) == {"AAPL", "BRK.B", "NOCIK"}  # blank-ticker row dropped
    assert by_ticker["AAPL"].cik == "0000320193"  # zero-padded to 10 digits
    assert by_ticker["BRK.B"].ticker == "BRK.B"  # normalized upper, punctuation kept
    assert by_ticker["NOCIK"].cik == ""  # missing CIK flagged empty, never faked


def test_resolve_constituents_handles_class_shares_dupes_and_unresolved() -> None:
    ticker_map = {
        "AAPL": CompanyRecord(ticker="AAPL", cik="0000320193", company_name="Apple Inc."),
        "BRK-B": CompanyRecord(ticker="BRK-B", cik="0001067983", company_name="Berkshire"),
    }

    result = resolve_constituents(["AAPL", "BRK.B", "AAPL", "ZZZZ"], ticker_map)

    resolved = {r.ticker for r in result.resolved}
    assert resolved == {"AAPL", "BRK-B"}  # BRK.B resolved via the dot->dash fallback
    assert result.unresolved == ("ZZZZ",)  # unknown flagged, not silently dropped
    assert len(result.resolved) == 2  # duplicate AAPL collapsed


def test_committed_sp500_csv_is_well_formed() -> None:
    records = parse_constituents_csv(CONSTITUENTS_CSV.read_text(encoding="utf-8"))

    assert len(records) >= 500  # the full index, not a stub
    tickers = [r.ticker for r in records]
    assert len(tickers) == len(set(tickers))  # no duplicate tickers
    for record in records:
        assert record.cik.isdigit() and len(record.cik) == 10, record.ticker
        assert record.company_name  # every row carries a name


# --------------------------------------------------------------------------- #
# Mocked SEC client + a representative per-company ingest
# --------------------------------------------------------------------------- #

def _submissions(cik: str) -> dict:
    """One 10-K filing payload keyed for `recent_filing_records`."""

    return {
        "name": f"COMPANY {cik}",
        "filings": {
            "recent": {
                "form": ["10-K"],
                "accessionNumber": [f"{cik}-26-000001"],
                "filingDate": ["2026-02-15"],
                "reportDate": ["2025-12-31"],
                "primaryDocument": ["primary.htm"],
                "primaryDocDescription": ["10-K"],
                "acceptanceDateTime": ["2026-02-15T16:00:00.000Z"],
            }
        },
    }


class FakeSECClient:
    """Records every network touch so tests can assert no re-fetch / no live call."""

    def __init__(self, ciks: list[str]) -> None:
        self._submissions = {cik: _submissions(cik) for cik in ciks}
        self.submission_calls: list[str] = []
        self.document_calls: list[str] = []

    def fetch_company_submissions(self, cik: str) -> dict:
        self.submission_calls.append(str(cik))
        return self._submissions[str(cik)]

    def fetch_document(self, url: str) -> bytes:
        self.document_calls.append(url)
        return b"<html><body>" + b"filing body text " * 300 + b"</body></html>"


def _company(ticker: str, cik: str) -> CompanyRecord:
    return CompanyRecord(ticker=ticker, cik=cik, company_name=f"{ticker} Inc.")


def _make_ingest(client: FakeSECClient, store: LocalRagStore, *, fail_on: set[str]):
    """Representative ingest: fetch submissions + docs, write raw idempotently."""

    def ingest_company(company: CompanyRecord) -> list[str]:
        if company.ticker in fail_on:
            raise RuntimeError(f"interrupted on {company.ticker}")
        submissions = client.fetch_company_submissions(company.cik)
        written: list[str] = []
        for record in recent_filing_records(submissions):
            raw_path = store.raw_path(
                company.ticker, record.accession_number, record.primary_document
            )
            if not raw_path.exists():
                content = client.fetch_document(f"https://sec/{record.accession_number}")
                store.write_bytes_once(raw_path, content)
            written.append(record.accession_number)
        return written

    return ingest_company


def _raw_file_count(store: LocalRagStore) -> int:
    return sum(1 for _ in store.raw_dir.rglob("*") if _.is_file())


def test_resume_after_interruption_skips_done_reruns_interrupted_no_dupes(
    tmp_path: Path,
) -> None:
    store = LocalRagStore(root=tmp_path)
    checkpoint = tmp_path / "checkpoint.jsonl"
    constituents = [_company("A", "1"), _company("B", "2"), _company("C", "3")]

    # First run interrupts on B (hard crash: continue_on_error=False propagates).
    client1 = FakeSECClient(["1", "2", "3"])
    with pytest.raises(RuntimeError):
        fetch_constituents(
            constituents=constituents,
            store=store,
            checkpoint_path=checkpoint,
            ingest_company=_make_ingest(client1, store, fail_on={"B"}),
            continue_on_error=False,
            logger=lambda _msg: None,
        )

    assert load_completed_tickers(checkpoint) == {"A"}  # only the pre-crash name
    assert client1.document_calls == ["https://sec/1-26-000001"]  # C never reached
    assert _raw_file_count(store) == 1

    # Re-run: A skipped (no network touch), B re-runs, C runs. No dupes.
    client2 = FakeSECClient(["1", "2", "3"])
    result = fetch_constituents(
        constituents=constituents,
        store=store,
        checkpoint_path=checkpoint,
        ingest_company=_make_ingest(client2, store, fail_on=set()),
        continue_on_error=False,
        logger=lambda _msg: None,
    )

    assert result.skipped == 1 and result.completed == 2
    assert "1" not in client2.submission_calls  # completed A not re-fetched
    assert client2.submission_calls == ["2", "3"]
    assert load_completed_tickers(checkpoint) == {"A", "B", "C"}
    assert _raw_file_count(store) == 3  # one raw doc per name, no duplicates


def test_idempotent_rerun_is_a_noop(tmp_path: Path) -> None:
    store = LocalRagStore(root=tmp_path)
    checkpoint = tmp_path / "checkpoint.jsonl"
    constituents = [_company("A", "1"), _company("B", "2")]

    client1 = FakeSECClient(["1", "2"])
    first = fetch_constituents(
        constituents=constituents,
        store=store,
        checkpoint_path=checkpoint,
        ingest_company=_make_ingest(client1, store, fail_on=set()),
        logger=lambda _msg: None,
    )
    assert first.completed == 2
    checkpoint_after_first = checkpoint.read_text(encoding="utf-8")

    # Second run touches nothing: all skipped, no network, checkpoint byte-identical.
    client2 = FakeSECClient(["1", "2"])
    second = fetch_constituents(
        constituents=constituents,
        store=store,
        checkpoint_path=checkpoint,
        ingest_company=_make_ingest(client2, store, fail_on=set()),
        logger=lambda _msg: None,
    )

    assert second.skipped == 2
    assert second.completed == 0 and second.chunks_written == 0
    assert client2.submission_calls == [] and client2.document_calls == []
    assert checkpoint.read_text(encoding="utf-8") == checkpoint_after_first


def test_continue_on_error_records_failure_and_keeps_going(tmp_path: Path) -> None:
    store = LocalRagStore(root=tmp_path)
    checkpoint = tmp_path / "checkpoint.jsonl"
    constituents = [_company("A", "1"), _company("B", "2"), _company("C", "3")]

    client = FakeSECClient(["1", "2", "3"])
    result = fetch_constituents(
        constituents=constituents,
        store=store,
        checkpoint_path=checkpoint,
        ingest_company=_make_ingest(client, store, fail_on={"B"}),
        continue_on_error=True,
        logger=lambda _msg: None,
    )

    assert result.completed == 2 and result.failed == 1
    assert result.failures and result.failures[0][0] == "B"
    assert load_completed_tickers(checkpoint) == {"A", "C"}  # B not checkpointed, retries later


# --------------------------------------------------------------------------- #
# The production script's ingest path runs on a mocked client (no live EDGAR)
# --------------------------------------------------------------------------- #

def _load_fetch_script():
    spec = importlib.util.spec_from_file_location(
        "financial_rag_sp500_fetch",
        REPO_ROOT / "scripts" / "financial_rag_sp500_fetch.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_ingest_company_wires_engine_on_mock_client(tmp_path: Path) -> None:
    module = _load_fetch_script()
    store = LocalRagStore(root=tmp_path)
    client = FakeSECClient(["1"])
    ingest_company = module.make_ingest_company(
        client=client,
        store=store,
        recent_8k_limit=0,  # no 8-K -> no filing-index fetch, keeps the mock minimal
        embed_batch_size=8,
        skip_embeddings=True,
    )

    chunks = ingest_company(_company("A", "1"))

    assert chunks and all(chunk.ticker == "A" for chunk in chunks)
    assert client.document_calls  # fetched through the real engine, mock client only
    fetch_count = len(client.document_calls)

    # Re-running the same name re-fetches nothing (write-once idempotency holds).
    ingest_company(_company("A", "1"))
    assert len(client.document_calls) == fetch_count
