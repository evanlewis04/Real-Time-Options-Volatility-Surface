"""Research-only local volatility-surface denoisers.

These helpers produce smooth estimates with explicit uncertainty and
provenance. They are opt-in research tools and are not market-truth sources.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.surface_features import SurfaceFeatureFrame, SurfaceFeatureSchema


DENOISED_PROVENANCE = "ml_denoised_research_estimate_not_market_observation"
ML_MODE_STATUS_OFF = "research_off_by_default"


@dataclass(frozen=True)
class SurfaceDenoiserResult:
    """Predicted IV estimates plus uncertainty and provenance metadata."""

    predictions: pd.Series
    uncertainty: pd.Series
    rows: pd.DataFrame
    metadata: dict[str, Any]

    def records(self) -> list[dict[str, Any]]:
        """Return deterministic dashboard/export records."""
        if self.predictions.empty:
            return []
        records = self.rows.copy()
        records["ml_denoised_iv"] = self.predictions.to_numpy(dtype=float)
        records["ml_uncertainty"] = self.uncertainty.to_numpy(dtype=float)
        records["provenance"] = DENOISED_PROVENANCE
        columns = [
            column
            for column in (
                "strike",
                "dte",
                "log_moneyness",
                "moneyness",
                "option_type",
                "target_iv",
                "ml_denoised_iv",
                "ml_uncertainty",
                "fit_weight",
                "provenance",
            )
            if column in records
        ]
        return records[columns].replace({np.nan: None}).to_dict("records")


@dataclass
class SurfaceDenoiserArtifact:
    """Persistable local denoiser with schema and research provenance."""

    model: Any
    schema: SurfaceFeatureSchema
    metadata: dict[str, Any]
    validation_metrics: dict[str, Any]
    feature_importances: dict[str, float] = field(default_factory=dict)

    def predict(self, feature_frame: SurfaceFeatureFrame) -> SurfaceDenoiserResult:
        """Predict denoised IV and uncertainty after schema compatibility checks."""
        _assert_schema_compatible(self.schema, feature_frame.schema)
        if feature_frame.empty():
            return SurfaceDenoiserResult(
                predictions=pd.Series(dtype=float, name="ml_denoised_iv"),
                uncertainty=pd.Series(dtype=float, name="ml_uncertainty"),
                rows=feature_frame.rows,
                metadata=_prediction_metadata(self.metadata, 0),
            )
        x = feature_frame.features.loc[:, self.schema.feature_names]
        predictions = np.clip(np.asarray(self.model.predict(x), dtype=float), 0.01, 5.0)
        uncertainty = _model_uncertainty(self.model, x, fallback=self.metadata.get("training_residual_std"))
        return SurfaceDenoiserResult(
            predictions=pd.Series(predictions, name="ml_denoised_iv"),
            uncertainty=pd.Series(uncertainty, name="ml_uncertainty"),
            rows=feature_frame.rows.reset_index(drop=True),
            metadata=_prediction_metadata(self.metadata, len(predictions)),
        )


class SurfaceDenoiser:
    """Baseline ExtraTrees surface denoiser trained only on provided local data."""

    def __init__(self, *, random_state: int = 17, n_estimators: int = 96, min_samples_leaf: int = 2):
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)

    def fit(
        self,
        feature_frame: SurfaceFeatureFrame,
        *,
        training_snapshot_range: tuple[str, str] | None = None,
    ) -> SurfaceDenoiserArtifact:
        """Fit a deterministic nonparametric model from a local feature frame."""
        if feature_frame.empty():
            raise ValueError("Cannot train surface denoiser without feature rows")
        regressor_cls = _extra_trees_regressor()
        x = feature_frame.features.loc[:, feature_frame.schema.feature_names]
        y = feature_frame.target.astype(float)
        weights = feature_frame.weights.astype(float).clip(lower=0.0)
        train_idx, valid_idx = _deterministic_validation_split(len(x))
        validation_metrics = _validation_metrics(
            regressor_cls,
            x,
            y,
            weights,
            train_idx,
            valid_idx,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
        )
        model = regressor_cls(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=1,
        )
        fit_kwargs = {"sample_weight": weights.to_numpy(dtype=float)} if np.any(weights > 0.0) else {}
        model.fit(x, y, **fit_kwargs)
        fitted = np.asarray(model.predict(x), dtype=float)
        residuals = fitted - y.to_numpy(dtype=float)
        metadata = {
            "model_family": "ExtraTreesRegressor",
            "mode": "ML Denoised",
            "status": "research_trained",
            "enabled_by_default": False,
            "training_rows": int(len(x)),
            "feature_schema_version": feature_frame.schema.version,
            "feature_count": int(len(feature_frame.schema.feature_names)),
            "training_snapshot_range": list(training_snapshot_range) if training_snapshot_range else None,
            "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "min_samples_leaf": self.min_samples_leaf,
            "training_residual_std": _std_or_zero(residuals),
            "provenance": DENOISED_PROVENANCE,
            "estimate_warning": "ML denoised values are research estimates, not market observations.",
        }
        return SurfaceDenoiserArtifact(
            model=model,
            schema=feature_frame.schema,
            metadata=metadata,
            validation_metrics=validation_metrics,
            feature_importances=_feature_importances(model, feature_frame.schema.feature_names),
        )


@dataclass
class SurfaceKernelSmoother:
    """Optional kernel smoother research mode with coordinate uncertainty."""

    bandwidth_log_moneyness: float = 0.08
    bandwidth_dte: float = 45.0

    def fit_predict(self, feature_frame: SurfaceFeatureFrame) -> SurfaceDenoiserResult:
        """Smooth IV over log-moneyness and DTE without external state."""
        if feature_frame.empty():
            return SurfaceDenoiserResult(
                predictions=pd.Series(dtype=float, name="ml_denoised_iv"),
                uncertainty=pd.Series(dtype=float, name="ml_uncertainty"),
                rows=feature_frame.rows,
                metadata=_kernel_metadata(0, self.bandwidth_log_moneyness, self.bandwidth_dte),
            )
        coords = feature_frame.rows[["log_moneyness", "dte"]].to_numpy(dtype=float)
        y = feature_frame.target.to_numpy(dtype=float)
        row_weights = feature_frame.weights.to_numpy(dtype=float)
        predictions: list[float] = []
        uncertainties: list[float] = []
        for coord in coords:
            scaled = ((coords[:, 0] - coord[0]) / self.bandwidth_log_moneyness) ** 2 + (
                (coords[:, 1] - coord[1]) / self.bandwidth_dte
            ) ** 2
            weights = np.exp(-0.5 * scaled) * np.clip(row_weights, 0.0, None)
            if not np.any(weights > 0.0):
                weights = np.exp(-0.5 * scaled)
            total = float(np.sum(weights))
            prediction = float(np.sum(weights * y) / total)
            local_var = float(np.sum(weights * (y - prediction) ** 2) / total)
            effective_n = (total**2) / max(float(np.sum(weights**2)), 1e-12)
            uncertainty = np.sqrt(max(local_var, 0.0)) + (1.0 / np.sqrt(max(effective_n, 1.0))) * 0.005
            predictions.append(float(np.clip(prediction, 0.01, 5.0)))
            uncertainties.append(float(uncertainty))
        return SurfaceDenoiserResult(
            predictions=pd.Series(predictions, name="ml_denoised_iv"),
            uncertainty=pd.Series(uncertainties, name="ml_uncertainty"),
            rows=feature_frame.rows.reset_index(drop=True),
            metadata=_kernel_metadata(len(predictions), self.bandwidth_log_moneyness, self.bandwidth_dte),
        )


def save_surface_denoiser(artifact: SurfaceDenoiserArtifact, directory: Path | str) -> Path:
    """Persist model bytes and JSON metadata for local experiments."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "surface_denoiser.pkl"
    metadata_path = root / "surface_denoiser.metadata.json"
    with model_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    metadata = {
        "schema": artifact.schema.as_dict(),
        "model_metadata": artifact.metadata,
        "validation_metrics": artifact.validation_metrics,
        "feature_importances": artifact.feature_importances,
        "provenance": DENOISED_PROVENANCE,
        "model_path": model_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def load_surface_denoiser(
    directory: Path | str,
    *,
    expected_schema: SurfaceFeatureSchema | None = None,
) -> SurfaceDenoiserArtifact:
    """Load a saved denoiser and refuse incompatible feature schemas."""
    root = Path(directory)
    model_path = root / "surface_denoiser.pkl"
    with model_path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, SurfaceDenoiserArtifact):
        raise TypeError("Saved surface denoiser payload is incompatible")
    if expected_schema is not None:
        _assert_schema_compatible(expected_schema, artifact.schema)
    return artifact


def ml_surface_mode_metadata(*, enabled: bool = False) -> dict[str, Any]:
    """Return explicit ML mode metadata while keeping the mode off by default."""
    return {
        "mode": "ML Denoised",
        "model": "ExtraTreesRegressor",
        "status": "available" if enabled else ML_MODE_STATUS_OFF,
        "enabled": bool(enabled),
        "enabled_by_default": False,
        "fit_policy": "local_research_denoiser_opt_in",
        "weighted_rmse": None,
        "rmse": None,
        "uncertainty": None,
        "provenance": DENOISED_PROVENANCE,
        "estimate_warning": "ML denoised values are estimates for research comparison, not market observations.",
    }


def _extra_trees_regressor() -> Any:
    try:
        from sklearn.ensemble import ExtraTreesRegressor
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for the baseline surface denoiser") from exc
    return ExtraTreesRegressor


def _deterministic_validation_split(size: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(size)
    if size < 10:
        return indices, np.array([], dtype=int)
    valid = indices[::5]
    train = np.array([idx for idx in indices if idx not in set(valid)], dtype=int)
    if len(train) < 5:
        return indices, np.array([], dtype=int)
    return train, valid


def _validation_metrics(
    regressor_cls: Any,
    x: pd.DataFrame,
    y: pd.Series,
    weights: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    *,
    random_state: int,
    n_estimators: int,
    min_samples_leaf: int,
) -> dict[str, Any]:
    if len(valid_idx) == 0:
        return {
            "method": "deterministic_holdout_every_5th_row",
            "validation_rows": 0,
            "rmse": None,
            "mae": None,
            "weighted_rmse": None,
        }
    model = regressor_cls(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
        n_jobs=1,
    )
    train_weights = weights.iloc[train_idx].to_numpy(dtype=float)
    fit_kwargs = {"sample_weight": train_weights} if np.any(train_weights > 0.0) else {}
    model.fit(x.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)
    predicted = np.asarray(model.predict(x.iloc[valid_idx]), dtype=float)
    actual = y.iloc[valid_idx].to_numpy(dtype=float)
    residuals = predicted - actual
    valid_weights = weights.iloc[valid_idx].to_numpy(dtype=float)
    return {
        "method": "deterministic_holdout_every_5th_row",
        "validation_rows": int(len(valid_idx)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "weighted_rmse": _weighted_rmse(residuals, valid_weights),
    }


def _model_uncertainty(model: Any, x: pd.DataFrame, *, fallback: Any) -> np.ndarray:
    estimators = getattr(model, "estimators_", None)
    if estimators:
        values = x.to_numpy(dtype=float)
        tree_predictions = np.vstack([estimator.predict(values) for estimator in estimators])
        uncertainty = np.std(tree_predictions, axis=0, ddof=0)
    else:
        fallback_value = float(fallback) if fallback is not None else 0.0
        uncertainty = np.full(len(x), fallback_value, dtype=float)
    return np.maximum(uncertainty.astype(float), 0.0)


def _prediction_metadata(model_metadata: dict[str, Any], row_count: int) -> dict[str, Any]:
    return {
        "mode": "ML Denoised",
        "status": "research_prediction",
        "row_count": int(row_count),
        "model_family": model_metadata.get("model_family"),
        "feature_schema_version": model_metadata.get("feature_schema_version"),
        "provenance": DENOISED_PROVENANCE,
        "estimate_warning": "ML denoised values are research estimates, not market observations.",
    }


def _kernel_metadata(row_count: int, bandwidth_log_moneyness: float, bandwidth_dte: float) -> dict[str, Any]:
    return {
        "mode": "ML Denoised",
        "model_family": "kernel_smoother",
        "status": "research_prediction",
        "row_count": int(row_count),
        "bandwidth_log_moneyness": float(bandwidth_log_moneyness),
        "bandwidth_dte": float(bandwidth_dte),
        "provenance": DENOISED_PROVENANCE,
        "estimate_warning": "Kernel-smoothed values are research estimates, not market observations.",
    }


def _feature_importances(model: Any, feature_names: tuple[str, ...]) -> dict[str, float]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {}
    return {
        name: float(value)
        for name, value in sorted(zip(feature_names, values), key=lambda item: item[0])
        if np.isfinite(float(value))
    }


def _weighted_rmse(residuals: np.ndarray, weights: np.ndarray) -> float:
    clean = np.asarray(weights, dtype=float)
    clean = np.where(np.isfinite(clean), clean, 0.0)
    clean = np.clip(clean, 0.0, None)
    if not np.any(clean > 0.0):
        return float(np.sqrt(np.mean(residuals**2)))
    return float(np.sqrt(np.average(residuals**2, weights=clean)))


def _std_or_zero(values: np.ndarray) -> float:
    std = float(np.std(np.asarray(values, dtype=float), ddof=0))
    return std if np.isfinite(std) else 0.0


def _assert_schema_compatible(expected: SurfaceFeatureSchema, actual: SurfaceFeatureSchema) -> None:
    if expected.version != actual.version or expected.feature_names != actual.feature_names:
        raise ValueError(
            "Surface denoiser feature schema mismatch: "
            f"expected {expected.version} with {len(expected.feature_names)} features, "
            f"got {actual.version} with {len(actual.feature_names)} features"
        )
