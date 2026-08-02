"""Regenerate the committed S&P 500 constituents CSV (``config/sp500_constituents.csv``).

Resolves a current-membership ticker list to CIKs against SEC's
``company_tickers.json`` (the same map ``sec_client`` uses) and writes the static
``ticker,cik,company_name`` CSV the Stage 4 fetch reads. Run this only to refresh
membership; the fetch itself never does a live lookup.

Membership source defaults to the public S&P 500 constituents dataset; override
with ``--membership-csv`` to point at a local CSV that has a ``Symbol`` column.
Resolution requires ``SEC_USER_AGENT`` (fair-access contact), exactly like the
real fetch. Any ticker that does not resolve is reported and left out — never
written with a guessed CIK (data-honesty guardrail).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.financial_rag.ingestion.constituents import (
    format_constituents_csv,
    resolve_constituents,
)
from src.financial_rag.ingestion.sec_client import SECClient
from src.financial_rag.settings import configured_secret, load_environment, project_root

DEFAULT_MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)
DEFAULT_OUTPUT = Path("config") / "sp500_constituents.csv"


def read_membership_tickers(text: str) -> list[str]:
    """Return the ``Symbol`` column from an S&P 500 constituents CSV."""

    reader = csv.DictReader(io.StringIO(text))
    return [str(row.get("Symbol", "")).strip() for row in reader if row.get("Symbol")]


def load_membership_text(*, membership_csv: str | None, user_agent: str) -> str:
    if membership_csv:
        return Path(membership_csv).read_text(encoding="utf-8")
    response = requests.get(
        DEFAULT_MEMBERSHIP_URL, headers={"User-Agent": user_agent}, timeout=30
    )
    response.raise_for_status()
    return response.text


def build_header(count: int) -> list[str]:
    today = datetime.date.today().isoformat()
    return [
        "S&P 500 constituents -- ticker,cik,company_name (Phase 1 Stage 4).",
        f"Static CURRENT membership snapshot as of {today} ({count} names). NOT",
        "point-in-time/historical membership: index adds/drops over time are not",
        "tracked (survivorship-bias limitation).",
        "Membership source: github.com/datasets/s-and-p-500-companies (constituents.csv).",
        "CIKs resolved against SEC company_tickers.json (the map sec_client uses).",
        "Regenerate with scripts/financial_rag_resolve_sp500.py.",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--membership-csv",
        default=None,
        help="Local CSV with a Symbol column. Default: fetch the public S&P 500 dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_environment()
    sec_user_agent = configured_secret("SEC_USER_AGENT")
    if not sec_user_agent:
        raise SystemExit(
            "SEC_USER_AGENT must be configured in .env to resolve CIKs from SEC EDGAR."
        )

    tickers = read_membership_tickers(
        load_membership_text(membership_csv=args.membership_csv, user_agent=sec_user_agent)
    )
    client = SECClient(user_agent=sec_user_agent)
    ticker_map = client.fetch_company_tickers()
    result = resolve_constituents(tickers, ticker_map)

    output_path = project_root() / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_constituents_csv(result.resolved, header_comment=build_header(len(result.resolved))),
        encoding="utf-8",
    )

    print(f"Membership tickers read: {len(tickers)}")
    print(f"Resolved to CIK: {len(result.resolved)}")
    print(f"Unresolved (omitted, not faked): {len(result.unresolved)}")
    if result.unresolved:
        print("  " + ", ".join(result.unresolved))
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
