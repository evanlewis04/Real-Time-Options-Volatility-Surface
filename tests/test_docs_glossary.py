from pathlib import Path


REQUIRED_TERMS = [
    "IV",
    "DTE",
    "Moneyness",
    "Log-Moneyness",
    "Delta",
    "Skew",
    "Risk Reversal",
    "Butterfly",
    "IV Rank",
    "SVI",
]


def test_glossary_documents_required_phase5_terms():
    glossary = Path("docs/glossary.md").read_text(encoding="utf-8")

    for term in REQUIRED_TERMS:
        assert f"**{term}**" in glossary
    assert "Synthetic" in glossary
    assert "Fallback" in glossary
