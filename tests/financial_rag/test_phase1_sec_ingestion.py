from src.financial_rag.ingestion.sec_client import (
    FilingRecord,
    accession_directory,
    build_document_id,
    cik_archive_value,
    cik_padded,
    filing_index_url,
    filing_primary_document_url,
    parse_filing_index_exhibits,
    recent_filing_records,
    sec_archive_base_url,
    select_phase1_filings,
)


def test_sec_url_and_accession_helpers_are_stable() -> None:
    filing = FilingRecord(
        form="10-K",
        accession_number="0000320193-25-000079",
        filing_date="2025-10-31",
        report_date="2025-09-27",
        primary_document="aapl-20250927.htm",
        description="10-K",
    )

    assert cik_padded(320193) == "0000320193"
    assert cik_archive_value("0000320193") == "320193"
    assert accession_directory(filing.accession_number) == "000032019325000079"
    assert sec_archive_base_url("0000320193", filing.accession_number) == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/"
    )
    assert filing_index_url("0000320193", filing.accession_number).endswith(
        "/0000320193-25-000079-index.html"
    )
    assert filing_primary_document_url("0000320193", filing).endswith(
        "/aapl-20250927.htm"
    )


def test_build_document_id_changes_by_role_and_document_name() -> None:
    primary = build_document_id(
        ticker="nvda",
        accession_number="0001045810-26-000051",
        document_role="primary",
        document_name="nvda-20260520.htm",
    )
    exhibit = build_document_id(
        ticker="nvda",
        accession_number="0001045810-26-000051",
        document_role="exhibit",
        document_name="q1fy27pr.htm",
    )

    assert primary.startswith("NVDA-000104581026000051")
    assert exhibit.startswith("NVDA-000104581026000051")
    assert primary != exhibit


def test_phase1_filing_selection_uses_exact_forms() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q/A", "10-Q", "10-K", "8-K", "10-K/A"],
                "accessionNumber": ["8k-1", "10qa", "10q", "10k", "8k-2", "10ka"],
                "filingDate": [
                    "2026-05-20",
                    "2026-05-10",
                    "2026-05-01",
                    "2026-02-01",
                    "2026-01-01",
                    "2025-12-01",
                ],
                "reportDate": ["", "", "2026-04-30", "2026-01-31", "", ""],
                "primaryDocument": ["a.htm", "qa.htm", "q.htm", "k.htm", "b.htm", "ka.htm"],
                "primaryDocDescription": ["8-K", "10-Q/A", "10-Q", "10-K", "8-K", "10-K/A"],
            }
        }
    }

    selection = select_phase1_filings(recent_filing_records(submissions), recent_8k_limit=2)

    assert selection.latest_10q is not None
    assert selection.latest_10q.accession_number == "10q"
    assert selection.latest_10k is not None
    assert selection.latest_10k.accession_number == "10k"
    assert [filing.accession_number for filing in selection.recent_8ks] == ["8k-1", "8k-2"]


def test_parse_filing_index_exhibits_discovers_ex99_html_and_skips_assets() -> None:
    html = """
    <table>
      <tr><td>1</td><td>FORM 8-K</td><td><a href="main.htm">main.htm</a></td><td>8-K</td></tr>
      <tr><td>2</td><td>Press Release</td><td><a href="q1pr.htm">q1pr.htm</a></td><td>EX-99.1</td></tr>
      <tr><td>3</td><td>EX-99.2</td><td><a href="/Archives/slides.htm">slides.htm</a></td><td>EX-99.2</td></tr>
      <tr><td>4</td><td>EX99 image</td><td><a href="chart.jpg">chart.jpg</a></td><td>GRAPHIC</td></tr>
    </table>
    """

    exhibits = parse_filing_index_exhibits(
        html,
        base_url="https://www.sec.gov/Archives/edgar/data/1/2/",
    )

    assert [exhibit.document for exhibit in exhibits] == ["q1pr.htm", "slides.htm"]
    assert exhibits[0].url == "https://www.sec.gov/Archives/edgar/data/1/2/q1pr.htm"
    assert exhibits[1].url == "https://www.sec.gov/Archives/slides.htm"
