"""Formatting helpers for dashboard display values."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def fmt_money(value: Optional[float]) -> str:
    return "n/a" if value is None or pd.isna(value) else f"${value:,.2f}"


def fmt_int(value: Optional[float]) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{int(value):,}"


def fmt_pct(value: Optional[float], digits: int = 1) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}%}"
