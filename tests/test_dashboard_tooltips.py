from src.dashboard.tooltips import COLUMN_HELP, CONTROL_HELP, KPI_HELP, column_help


def test_greek_columns_have_compact_tooltips():
    for column in ("Delta", "Gamma", "Theta/day", "Vega/1%"):
        text = COLUMN_HELP[column]
        assert text
        assert len(text) <= 80


def test_core_controls_and_kpis_have_tooltips():
    for key in ("universe", "primary_underlying", "max_spread_pct", "min_open_interest"):
        assert CONTROL_HELP[key]

    for key in ("Spot", "ATM IV", "Term Spread", "Median Spread"):
        assert KPI_HELP[key]


def test_column_help_returns_known_help_or_none():
    assert column_help("impliedVolatility") == COLUMN_HELP["impliedVolatility"]
    assert column_help("not-a-dashboard-column") is None
