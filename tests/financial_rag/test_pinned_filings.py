"""Guard the accession-pin reproducibility + entity-correctness contract.

These pins are the single source of truth for the filings the gold eval depends
on. This test enforces that each pin (a) derives to the correct filer CIK from
its own accession number, (b) is wired into the repair fetch path, and (c) when a
built corpus is present, resolves under that accession-derived CIK tagged to the
intended ticker -- so a future latest-N drift or a decoy ticker->CIK mapping
cannot silently reopen the gaps or re-contaminate a ticker with another
registrant's filings.
"""

from __future__ import annotations

import json

import pytest

from src.financial_rag.ingestion.pinned_filings import (
    PINNED_FILINGS,
    cik_from_accession,
    pins_for_tickers,
)
from src.financial_rag.settings import project_root
from src.financial_rag.storage.local_store import LocalRagStore, read_jsonl


# Intended filer CIK for each pinned accession, asserted independently of the
# derivation so the test pins the *entity*, not just the arithmetic. XOM's is the
# operating registrant Exxon Mobil Corp (34088), deliberately NOT the XOM ticker
# map's CIK 2115436 (ExxonMobil Holdings Corp, the 2026 holdco parent).
EXPECTED_PIN_CIK = {
    "0001045810-26-000051": 1045810,  # NVDA earnings 8-K
    "0000034088-26-000045": 34088,  # Exxon Mobil Corp 10-K
}


def test_cik_from_accession_extracts_the_filer_prefix() -> None:
    assert cik_from_accession("0000034088-26-000045") == 34088
    assert cik_from_accession("0001045810-26-000051") == 1045810
    with pytest.raises(ValueError):
        cik_from_accession("not-an-accession")


def test_every_pin_derives_to_its_intended_registrant_cik() -> None:
    for pin in PINNED_FILINGS:
        assert pin.accession_number in EXPECTED_PIN_CIK, (
            f"Pin {pin.accession_number} lacks an intended-CIK assertion; add one so "
            "the entity it fetches under is guarded."
        )
        assert cik_from_accession(pin.accession_number) == EXPECTED_PIN_CIK[pin.accession_number]


def test_pins_are_selected_by_ticker_for_the_repair_fetch_path() -> None:
    # This is exactly how the repair script picks which pins to fetch; guarding it
    # keeps the pin list from becoming a comment nobody reads.
    tickers = {pin.ticker for pin in PINNED_FILINGS}
    for pin in PINNED_FILINGS:
        selected = pins_for_tickers([pin.ticker])
        assert pin in selected
    assert set(pins_for_tickers(tickers)) == set(PINNED_FILINGS)
    assert pins_for_tickers(["ZZZZ"]) == ()


def test_pinned_filings_resolve_in_the_built_corpus_under_the_right_entity() -> None:
    store = LocalRagStore(root=project_root())
    manifest = store.chunks_dir / "manifest.jsonl"
    if not manifest.exists():
        pytest.skip("No built corpus present; run the repair script with --fetch-sec first.")

    rows = list(read_jsonl(manifest))
    for pin in PINNED_FILINGS:
        matches = [
            row
            for row in rows
            if str(row.get("accession_number", "")) == pin.accession_number
        ]
        assert matches, f"Pinned accession {pin.accession_number} has no chunks in the corpus."
        for row in matches:
            assert str(row.get("ticker", "")).upper() == pin.ticker, (
                f"Pinned accession {pin.accession_number} is tagged "
                f"{row.get('ticker')!r}, expected {pin.ticker}."
            )
            assert int(str(row.get("cik", "0"))) == EXPECTED_PIN_CIK[pin.accession_number], (
                f"Pinned accession {pin.accession_number} resolved under CIK "
                f"{row.get('cik')!r}, expected {EXPECTED_PIN_CIK[pin.accession_number]}."
            )


def test_repair_manifest_json_is_wellformed() -> None:
    # Cheap belt-and-suspenders: the corpus manifest, when present, is valid JSONL.
    store = LocalRagStore(root=project_root())
    manifest = store.chunks_dir / "manifest.jsonl"
    if not manifest.exists():
        pytest.skip("No built corpus present.")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)
