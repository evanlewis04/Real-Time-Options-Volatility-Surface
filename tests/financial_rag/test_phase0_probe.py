from datetime import UTC, datetime

from scripts.financial_rag_phase0_probe import (
    AlphaVantageResult,
    FilingRecord,
    TickerProbeResult,
    accession_directory,
    cik_archive_value,
    cik_padded,
    company_ticker_map,
    filing_index_url,
    exhibit_description,
    latest_filing,
    parse_filing_index_exhibits,
    recent_filing_records,
    recent_filings,
    render_markdown_report,
    sec_archive_base_url,
    summarize_alpha_vantage_response,
)


def test_sec_identifier_helpers_build_expected_archive_urls() -> None:
    accession = "0000320193-25-000079"

    assert cik_padded(320193) == "0000320193"
    assert cik_archive_value("0000320193") == "320193"
    assert accession_directory(accession) == "000032019325000079"
    assert sec_archive_base_url("0000320193", accession) == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/"
    )
    assert filing_index_url("0000320193", accession).endswith(
        "/0000320193-25-000079-index.html"
    )


def test_company_ticker_map_normalizes_sec_payload() -> None:
    payload = {
        "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    }

    mapped = company_ticker_map(payload)

    assert mapped["AAPL"]["cik_str"] == 320193
    assert mapped["MSFT"]["title"] == "MICROSOFT CORP"


def test_recent_filing_selection_uses_exact_forms() -> None:
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q", "10-Q/A", "10-K", "8-K"],
                "accessionNumber": [
                    "0001-8k-a",
                    "0002-10q",
                    "0003-10qa",
                    "0004-10k",
                    "0005-8k-b",
                ],
                "filingDate": [
                    "2026-05-01",
                    "2026-04-25",
                    "2026-04-26",
                    "2026-02-20",
                    "2026-01-15",
                ],
                "reportDate": ["", "2026-03-31", "2026-03-31", "2025-12-31", ""],
                "primaryDocument": ["a.htm", "q.htm", "qa.htm", "k.htm", "b.htm"],
                "primaryDocDescription": ["8-K", "10-Q", "10-Q/A", "10-K", "8-K"],
            }
        }
    }

    records = recent_filing_records(submissions)

    assert latest_filing(records, "10-Q").accession_number == "0002-10q"
    assert latest_filing(records, "10-K").accession_number == "0004-10k"
    assert [record.accession_number for record in recent_filings(records, "8-K", limit=2)] == [
        "0001-8k-a",
        "0005-8k-b",
    ]


def test_parse_filing_index_exhibits_filters_ex99_rows() -> None:
    html = """
    <table>
      <tr>
        <td>1</td><td>FORM 8-K</td><td><a href="/Archives/main.htm">main.htm</a></td><td>8-K</td><td>1000</td>
      </tr>
      <tr>
        <td>2</td><td>Press Release</td><td><a href="ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>4200</td>
      </tr>
      <tr>
        <td>3</td><td>EX-99.2</td><td><a href="slides.htm">slides.htm</a></td><td>EX-99.2</td><td>9000</td>
      </tr>
      <tr>
        <td>4</td><td>Investor Presentation EX99</td><td><a href="slides-asset.jpg">slides-asset.jpg</a></td><td>GRAPHIC</td><td>9000</td>
      </tr>
    </table>
    """

    exhibits = parse_filing_index_exhibits(
        html,
        base_url="https://www.sec.gov/Archives/edgar/data/1/2/",
    )

    assert [exhibit.document for exhibit in exhibits] == ["ex991.htm", "slides.htm"]
    assert exhibits[0].url == "https://www.sec.gov/Archives/edgar/data/1/2/ex991.htm"
    assert exhibits[1].description == "EX-99.2"
    assert exhibit_description(exhibits[1]) == "EX-99.2; inferred: presentation slides"


def test_summarize_alpha_vantage_response_handles_common_shapes() -> None:
    assert summarize_alpha_vantage_response({"Error Message": "bad function"}).startswith("error:")
    assert summarize_alpha_vantage_response({"Information": "premium endpoint"}).startswith(
        "information:"
    )
    assert summarize_alpha_vantage_response({"Note": "rate limit"}).startswith(
        "rate_limit_or_note:"
    )
    assert summarize_alpha_vantage_response(
        {"transcript": [{"speaker": "CEO", "content": "Hello"}]}
    ) == "ok: transcript list with 1 turns; sample keys: ['content', 'speaker']"


def test_render_markdown_report_includes_required_phase0_sections() -> None:
    filing = FilingRecord(
        form="8-K",
        accession_number="0000320193-25-000079",
        filing_date="2026-05-01",
        report_date="2026-04-30",
        primary_document="aapl-20260501.htm",
        description="FORM 8-K",
    )
    result = TickerProbeResult(
        ticker="AAPL",
        company_name="Apple Inc.",
        cik="0000320193",
        cik_status="resolved",
        latest_10k=None,
        latest_10q=None,
        recent_8ks=(filing,),
        ex99_by_accession={filing.accession_number: ()},
    )

    report = render_markdown_report(
        run_at=datetime(2026, 5, 24, tzinfo=UTC),
        tickers=("AAPL",),
        sec_results=(result,),
        alpha_result=AlphaVantageResult(status="skipped", detail="missing key"),
    )

    assert "Date run: 2026-05-24 00:00:00 UTC" in report
    assert "## SEC CIK Resolution" in report
    assert "## SEC Filing Coverage" in report
    assert "## EX-99 Exhibit Coverage" in report
    assert "## Alpha Vantage Transcript Probe" in report
    assert "## Recommendation" in report
