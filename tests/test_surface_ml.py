import numpy as np
import pandas as pd
import pytest

from src.ml.surface_denoiser import (
    DENOISED_PROVENANCE,
    SurfaceDenoiser,
    SurfaceKernelSmoother,
    load_surface_denoiser,
    ml_surface_mode_metadata,
    save_surface_denoiser,
)
from src.ml.surface_features import (
    FEATURE_PROVENANCE,
    PRIOR_FEATURE_PROVENANCE,
    SurfaceFeatureSchema,
    build_surface_ml_features,
)
from src.quant.quote_quality import apply_quote_reliability_scores
from src.quant.surface_prior import HistoricalSurfacePrior
from tests.fixtures.noisy_option_chain import FIXTURE_SPOT, checked_clean_chain


def test_surface_ml_feature_set_is_stable_and_labels_prior_estimates():
    chain, meta = checked_clean_chain()
    chain, _ = apply_quote_reliability_scores(chain, meta)
    prior = _fixture_prior(chain)

    feature_frame = build_surface_ml_features(chain, FIXTURE_SPOT, prior=prior)

    assert feature_frame.schema.version == "surface_ml_features_v1"
    assert list(feature_frame.features.columns) == list(feature_frame.schema.feature_names)
    assert not feature_frame.features.isna().any().any()
    assert len(feature_frame.features) == len(feature_frame.target)
    assert feature_frame.metadata["provenance"] == FEATURE_PROVENANCE
    assert feature_frame.metadata["historical_prior_feature"]
    assert feature_frame.metadata["historical_prior_provenance"] == PRIOR_FEATURE_PROVENANCE
    assert "historical_iv_prior" in feature_frame.features
    assert feature_frame.features["historical_iv_prior"].gt(0.0).all()
    assert {"option_type_call", "option_type_put", "price_source_mark"}.issubset(feature_frame.features.columns)


def test_extra_trees_surface_denoiser_outputs_research_estimates_with_uncertainty():
    chain, meta = checked_clean_chain()
    chain, _ = apply_quote_reliability_scores(chain, meta)
    feature_frame = build_surface_ml_features(chain, FIXTURE_SPOT, prior=_fixture_prior(chain))

    artifact = SurfaceDenoiser(random_state=7, n_estimators=32, min_samples_leaf=1).fit(
        feature_frame,
        training_snapshot_range=("2026-05-01T10:00:00", "2026-05-08T10:00:00"),
    )
    result = artifact.predict(feature_frame)

    assert artifact.metadata["enabled_by_default"] is False
    assert artifact.metadata["provenance"] == DENOISED_PROVENANCE
    assert artifact.validation_metrics["method"] == "deterministic_holdout_every_5th_row"
    assert artifact.validation_metrics["validation_rows"] > 0
    assert result.metadata["provenance"] == DENOISED_PROVENANCE
    assert len(result.predictions) == len(feature_frame.target)
    assert np.max(np.abs(result.predictions - feature_frame.target)) < 0.02
    assert result.uncertainty.ge(0.0).all()
    assert result.records()[0]["provenance"] == DENOISED_PROVENANCE


def test_kernel_smoother_and_persistence_keep_schema_guardrails(tmp_path):
    chain, meta = checked_clean_chain()
    chain, _ = apply_quote_reliability_scores(chain, meta)
    feature_frame = build_surface_ml_features(chain, FIXTURE_SPOT, prior=_fixture_prior(chain))

    kernel_result = SurfaceKernelSmoother().fit_predict(feature_frame)
    assert kernel_result.metadata["model_family"] == "kernel_smoother"
    assert kernel_result.metadata["provenance"] == DENOISED_PROVENANCE
    assert kernel_result.uncertainty.ge(0.0).all()

    artifact = SurfaceDenoiser(random_state=11, n_estimators=16, min_samples_leaf=1).fit(feature_frame)
    metadata_path = save_surface_denoiser(artifact, tmp_path)
    assert metadata_path.exists()
    loaded = load_surface_denoiser(tmp_path, expected_schema=feature_frame.schema)
    loaded_result = loaded.predict(feature_frame)
    assert np.allclose(loaded_result.predictions, artifact.predict(feature_frame).predictions)

    incompatible_schema = SurfaceFeatureSchema(version="surface_ml_features_v2")
    with pytest.raises(ValueError, match="feature schema mismatch"):
        load_surface_denoiser(tmp_path, expected_schema=incompatible_schema)


def test_ml_surface_mode_metadata_is_explicitly_off_by_default():
    metadata = ml_surface_mode_metadata()

    assert metadata["mode"] == "ML Denoised"
    assert metadata["status"] == "research_off_by_default"
    assert metadata["enabled_by_default"] is False
    assert metadata["provenance"] == DENOISED_PROVENANCE


def _fixture_prior(chain: pd.DataFrame) -> HistoricalSurfacePrior:
    work = chain.copy()
    work["dte"] = pd.to_numeric(work["daysToExpiration"], errors="coerce")
    work["log_moneyness"] = pd.to_numeric(work["logMoneyness"], errors="coerce")
    work["prior_iv"] = pd.to_numeric(work["computedIV"], errors="coerce") - 0.005
    grid = (
        work.groupby(["dte", "log_moneyness"], as_index=False)["prior_iv"]
        .median()
        .assign(
            moneyness=lambda frame: np.exp(frame["log_moneyness"]),
            observations=2,
            snapshot_count=2,
            source="fixture",
            provenance=PRIOR_FEATURE_PROVENANCE,
        )
        .sort_values(["dte", "log_moneyness"])
        .reset_index(drop=True)
    )
    return HistoricalSurfacePrior(
        available=True,
        symbol="AAPL",
        source="fixture",
        grid=grid,
        snapshot_timestamps=("2026-05-01T10:00:00", "2026-05-02T10:00:00"),
        latest_snapshot_timestamp="2026-05-02T10:00:00",
        latest_age_days=1.0,
        snapshot_count=2,
        source_point_count=int(len(work)),
        cell_count=int(len(grid)),
        max_age_days=5.0,
    )
