"""Analytic validation utilities for SpatialSHAP research workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from spatialshap._geo_explainer import GeoExplainer
from spatialshap._geo_explanation import GeoExplanation
from spatialshap._references import KernelReference

FloatArray: TypeAlias = NDArray[np.float64]


def _readonly_float_array(values: Any, *, ndim: int, name: str) -> FloatArray:
    array: FloatArray = np.asarray(values, dtype=float).copy()
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RecoveryMetrics:
    """Scalar recovery statistics for one estimated component."""

    n_values: int
    rmse: float
    mae: float
    max_abs_error: float
    correlation: float
    sign_agreement: float

    def as_dict(self) -> dict[str, float | int]:
        """Return a flat representation suitable for result tables."""

        return {
            "n_values": self.n_values,
            "rmse": self.rmse,
            "mae": self.mae,
            "max_abs_error": self.max_abs_error,
            "correlation": self.correlation,
            "sign_agreement": self.sign_agreement,
        }


def recovery_metrics(
    estimated: Any,
    truth: Any,
    *,
    zero_tolerance: float = 1e-12,
) -> RecoveryMetrics:
    """Compare one estimated array with its known truth."""

    if not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("zero_tolerance must be finite and non-negative.")
    estimate: FloatArray = np.asarray(estimated, dtype=float)
    target: FloatArray = np.asarray(truth, dtype=float)
    if estimate.shape != target.shape:
        raise ValueError("estimated and truth must have the same shape.")
    if estimate.size == 0:
        raise ValueError("estimated and truth must not be empty.")
    if not np.isfinite(estimate).all() or not np.isfinite(target).all():
        raise ValueError("estimated and truth must contain only finite values.")

    estimate_flat = estimate.reshape(-1)
    target_flat = target.reshape(-1)
    error = estimate_flat - target_flat
    estimate_std = float(np.std(estimate_flat))
    target_std = float(np.std(target_flat))
    if estimate_std <= zero_tolerance and target_std <= zero_tolerance:
        correlation = (
            1.0
            if np.allclose(
                estimate_flat,
                target_flat,
                atol=zero_tolerance,
                rtol=0.0,
            )
            else float("nan")
        )
    elif estimate_std <= zero_tolerance or target_std <= zero_tolerance:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(estimate_flat, target_flat)[0, 1])

    estimate_sign = np.where(
        np.abs(estimate_flat) <= zero_tolerance,
        0.0,
        np.sign(estimate_flat),
    )
    target_sign = np.where(
        np.abs(target_flat) <= zero_tolerance,
        0.0,
        np.sign(target_flat),
    )
    return RecoveryMetrics(
        n_values=int(error.size),
        rmse=float(np.sqrt(np.mean(np.square(error)))),
        mae=float(np.mean(np.abs(error))),
        max_abs_error=float(np.max(np.abs(error))),
        correlation=correlation,
        sign_agreement=float(np.mean(estimate_sign == target_sign)),
    )


@dataclass(frozen=True)
class LinearReferenceSwitchTruth:
    """Analytic four-component truth for an additive linear prediction function."""

    primary_values: FloatArray
    geo_values: FloatArray
    interaction_values: FloatArray
    base_values: FloatArray
    predictions: FloatArray

    def __post_init__(self) -> None:
        primary = _readonly_float_array(
            self.primary_values,
            ndim=2,
            name="primary_values",
        )
        interaction = _readonly_float_array(
            self.interaction_values,
            ndim=2,
            name="interaction_values",
        )
        geo = _readonly_float_array(self.geo_values, ndim=1, name="geo_values")
        base = _readonly_float_array(self.base_values, ndim=1, name="base_values")
        predictions = _readonly_float_array(
            self.predictions,
            ndim=1,
            name="predictions",
        )
        n_rows, _ = primary.shape
        if interaction.shape != primary.shape:
            raise ValueError("interaction_values must match primary_values.")
        if geo.shape != (n_rows,) or base.shape != (n_rows,):
            raise ValueError("geo_values and base_values must match the row count.")
        if predictions.shape != (n_rows,):
            raise ValueError("predictions must match the row count.")
        object.__setattr__(self, "primary_values", primary)
        object.__setattr__(self, "interaction_values", interaction)
        object.__setattr__(self, "geo_values", geo)
        object.__setattr__(self, "base_values", base)
        object.__setattr__(self, "predictions", predictions)

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(n_observations, n_features)``."""

        return int(self.primary_values.shape[0]), int(self.primary_values.shape[1])

    @property
    def shapley_values(self) -> FloatArray:
        """Return analytic ordinary joint-game Shapley values."""

        feature = self.primary_values + 0.5 * self.interaction_values
        geo = self.geo_values + 0.5 * self.interaction_values.sum(axis=1)
        values: FloatArray = np.column_stack([feature, geo])
        values.setflags(write=False)
        return values

    @property
    def additivity_error(self) -> FloatArray:
        """Return analytic prediction reconstruction error."""

        reconstructed = (
            self.base_values
            + self.primary_values.sum(axis=1)
            + self.geo_values
            + self.interaction_values.sum(axis=1)
        )
        error: FloatArray = self.predictions - reconstructed
        error.setflags(write=False)
        return error


def _normalized_weights(
    values: Any,
    *,
    shape: tuple[int, ...],
    name: str,
) -> FloatArray:
    weights: FloatArray = np.asarray(values, dtype=float)
    if weights.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    if weights.ndim == 1:
        total = float(weights.sum())
        if total <= 0:
            raise ValueError(f"{name} must have positive total weight.")
        return np.asarray(weights / total, dtype=float)
    totals = weights.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError(f"Every row of {name} must have positive total weight.")
    return np.asarray(weights / totals, dtype=float)


def linear_reference_switch_truth(
    X: Any,
    coefficients: Any,
    *,
    background: Any,
    local_weights: Any,
    intercept: float = 0.0,
    global_weights: Any | None = None,
) -> LinearReferenceSwitchTruth:
    """Return the exact four-component truth for ``intercept + X @ coefficients``.

    The global reference mean defines feature primary contributions. Each focal
    local-reference mean produces a per-feature shift. The sum of those shifts is
    the GEO main term and their negatives are the GEO–feature interactions.
    """

    data = _readonly_float_array(X, ndim=2, name="X")
    reference = _readonly_float_array(background, ndim=2, name="background")
    coefficient = _readonly_float_array(
        coefficients,
        ndim=1,
        name="coefficients",
    )
    if data.shape[1] != reference.shape[1] or coefficient.shape != (data.shape[1],):
        raise ValueError("X, background, and coefficients must share a feature count.")
    if not np.isfinite(intercept):
        raise ValueError("intercept must be finite.")

    n_rows = int(data.shape[0])
    n_background = int(reference.shape[0])
    local = _normalized_weights(
        local_weights,
        shape=(n_rows, n_background),
        name="local_weights",
    )
    if global_weights is None:
        global_weight = np.full(n_background, 1.0 / n_background, dtype=float)
    else:
        global_weight = _normalized_weights(
            global_weights,
            shape=(n_background,),
            name="global_weights",
        )

    global_mean: FloatArray = global_weight @ reference
    local_means: FloatArray = local @ reference
    primary: FloatArray = (data - global_mean) * coefficient
    shifts: FloatArray = (local_means - global_mean) * coefficient
    geo: FloatArray = shifts.sum(axis=1)
    interactions: FloatArray = -shifts
    base_value = float(intercept + global_mean @ coefficient)
    base: FloatArray = np.full(n_rows, base_value, dtype=float)
    predictions: FloatArray = intercept + data @ coefficient
    return LinearReferenceSwitchTruth(
        primary_values=primary,
        geo_values=geo,
        interaction_values=interactions,
        base_values=base,
        predictions=predictions,
    )


def geo_recovery_table(
    explanation: GeoExplanation,
    truth: LinearReferenceSwitchTruth,
) -> pd.DataFrame:
    """Return recovery metrics for every geographic decomposition layer."""

    if explanation.shape != truth.shape:
        raise ValueError("explanation and truth must have the same shape.")
    comparisons = {
        "primary": recovery_metrics(
            explanation.primary_values,
            truth.primary_values,
        ),
        "geo_main": recovery_metrics(explanation.geo_values, truth.geo_values),
        "geo_interaction": recovery_metrics(
            explanation.interaction_values,
            truth.interaction_values,
        ),
        "joint_feature_shapley": recovery_metrics(
            explanation.shapley_values[:, :-1],
            truth.shapley_values[:, :-1],
        ),
        "joint_geo_shapley": recovery_metrics(
            explanation.shapley_values[:, -1],
            truth.shapley_values[:, -1],
        ),
    }
    table = pd.DataFrame(
        {name: metric.as_dict() for name, metric in comparisons.items()}
    ).T
    table.index.name = "component"
    return table


def bandwidth_sensitivity(
    model: Any,
    X: Any,
    *,
    background: Any,
    geometry: Any,
    background_geometry: Any,
    bandwidths: Sequence[float],
    kernel: str = "gaussian",
    max_exact_features: int = 11,
) -> pd.DataFrame:
    """Evaluate joint geographic magnitudes and diagnostics across bandwidths."""

    bandwidth_values = tuple(float(value) for value in bandwidths)
    if not bandwidth_values:
        raise ValueError("bandwidths must contain at least one value.")
    if any(not np.isfinite(value) or value <= 0 for value in bandwidth_values):
        raise ValueError("Every bandwidth must be finite and positive.")

    rows: list[dict[str, float]] = []
    for bandwidth in bandwidth_values:
        result = GeoExplainer(
            model,
            background,
            reference=KernelReference(bandwidth=bandwidth, kernel=kernel),
            background_geometry=background_geometry,
            max_exact_features=max_exact_features,
        )(X, geometry=geometry)
        effective = np.asarray(
            [item.effective_n for item in result.reference_diagnostics],
            dtype=float,
        )
        equivalence = np.asarray(
            [
                item.shapley_equivalence_error
                for item in result.decomposition_diagnostics
            ],
            dtype=float,
        )
        rows.append(
            {
                "bandwidth": bandwidth,
                "mean_abs_geo_main": float(np.mean(np.abs(result.geo_values))),
                "mean_abs_geo_interaction": float(
                    np.mean(np.abs(result.interaction_values))
                ),
                "mean_abs_joint_geo_shapley": float(
                    np.mean(np.abs(result.shapley_values[:, -1]))
                ),
                "mean_effective_reference_n": float(np.mean(effective)),
                "max_additivity_error": float(
                    np.max(np.abs(result.additivity_error))
                ),
                "max_shapley_equivalence_error": float(np.max(equivalence)),
            }
        )
    return pd.DataFrame(rows)
