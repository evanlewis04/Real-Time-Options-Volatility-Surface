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


def test_surface_quality_guide_documents_presets_and_provenance():
    guide = Path("docs/surface_quality.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    for phrase in (
        "Standard",
        "Robust",
        "Strict",
        "Diagnostic Raw",
        "Prior Assisted",
        "ML Denoised",
        "fit_mode_validation_diagnostic_not_market_observation",
        "prior_assisted_fit_estimate_not_market_observation",
        "ml_denoised_research_estimate_not_market_observation",
    ):
        assert phrase in guide
    assert "python scripts\\validate_surface_fit_modes.py --json" in guide
    assert "docs/surface_quality.md" in readme
