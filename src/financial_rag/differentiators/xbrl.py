"""Offline XBRL/companyfacts scaffolding for numeric checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class XbrlCheckResult:
    status: str
    ticker: str
    fact_name: str
    value: float | int | str | None = None
    unit: str = ""
    end_date: str = ""
    source_path: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_local_companyfacts(
    *,
    ticker: str,
    fact_name: str,
    facts_path: Path | str,
) -> XbrlCheckResult:
    """Read a local SEC companyfacts-style JSON file when available."""

    path = Path(facts_path)
    if not path.exists():
        return XbrlCheckResult(
            status="unavailable",
            ticker=ticker.upper(),
            fact_name=fact_name,
            source_path=str(path),
            message="Local companyfacts JSON is not available.",
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    fact = _find_fact(payload, fact_name)
    if fact is None:
        return XbrlCheckResult(
            status="not_found",
            ticker=ticker.upper(),
            fact_name=fact_name,
            source_path=str(path),
            message="Fact was not found in local companyfacts JSON.",
        )
    unit, values = next(iter(fact.get("units", {}).items()), ("", []))
    if not values:
        return XbrlCheckResult(
            status="not_found",
            ticker=ticker.upper(),
            fact_name=fact_name,
            unit=str(unit),
            source_path=str(path),
            message="Fact exists but has no unit values.",
        )
    latest = sorted(values, key=lambda item: str(item.get("end", "")))[-1]
    return XbrlCheckResult(
        status="ok",
        ticker=ticker.upper(),
        fact_name=fact_name,
        value=latest.get("val"),
        unit=str(unit),
        end_date=str(latest.get("end", "")),
        source_path=str(path),
        message="Loaded from local companyfacts JSON.",
    )


def _find_fact(payload: dict[str, Any], fact_name: str) -> dict[str, Any] | None:
    facts = payload.get("facts", {})
    for taxonomy in facts.values():
        if fact_name in taxonomy:
            return taxonomy[fact_name]
        for key, value in taxonomy.items():
            if key.lower() == fact_name.lower():
                return value
    return None
