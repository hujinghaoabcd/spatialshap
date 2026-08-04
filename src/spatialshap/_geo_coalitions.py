"""Exact joint geographic coalition evaluation and decomposition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from spatialshap._coalitions import exact_shapley_values

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class GeoDecompositionDiagnostics:
    """Numerical diagnostics for one exact joint geographic decomposition."""

    n_players: int
    n_coalitions: int
    design_rank: int
    condition_number: float
    constraint_error: float
    shapley_equivalence_error: float

    def as_dict(self) -> dict[str, float | int]:
        """Return flat fields suitable for tabular audit output."""

        return {
            "decomposition_n_players": self.n_players,
            "decomposition_n_coalitions": self.n_coalitions,
            "decomposition_design_rank": self.design_rank,
            "decomposition_condition_number": self.condition_number,
            "decomposition_constraint_error": self.constraint_error,
            "decomposition_shapley_equivalence_error": self.shapley_equivalence_error,
        }


def _validate_weights(
    weights: FloatArray,
    *,
    n_background: int,
    name: str,
) -> FloatArray:
    array: FloatArray = np.asarray(weights, dtype=float)
    if array.shape != (n_background,):
        raise ValueError(f"{name} must have shape ({n_background},).")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    total = float(array.sum())
    if total <= 0:
        raise ValueError(f"{name} must contain positive total weight.")
    return np.asarray(array / total, dtype=float)


def exact_joint_coalition_values(
    predict: Callable[[FloatArray], FloatArray],
    target: FloatArray,
    background: FloatArray,
    global_weights: FloatArray,
    local_weights: FloatArray,
) -> FloatArray:
    """Evaluate a game where GEO switches global weights to local weights.

    The GEO player is the last player. Model predictions are evaluated once per
    feature coalition and reused for the two reference distributions.
    """

    target_array: FloatArray = np.asarray(target, dtype=float)
    background_array: FloatArray = np.asarray(background, dtype=float)
    if target_array.ndim != 1:
        raise ValueError("target must be one-dimensional.")
    if background_array.ndim != 2 or background_array.shape[1] != target_array.size:
        raise ValueError("background must have shape (n_background, n_features).")
    n_background = int(background_array.shape[0])
    if n_background <= 0:
        raise ValueError("background must contain at least one row.")
    global_array = _validate_weights(
        global_weights,
        n_background=n_background,
        name="global_weights",
    )
    local_array = _validate_weights(
        local_weights,
        n_background=n_background,
        name="local_weights",
    )

    n_features = int(target_array.size)
    geo_bit = 1 << n_features
    values: FloatArray = np.empty(1 << (n_features + 1), dtype=float)
    for feature_mask in range(1 << n_features):
        mixed = background_array.copy()
        for feature in range(n_features):
            if feature_mask & (1 << feature):
                mixed[:, feature] = target_array[feature]
        predictions: FloatArray = np.asarray(predict(mixed), dtype=float)
        if predictions.shape != (n_background,):
            raise ValueError(
                "The prediction callable must return shape (n_samples,) for "
                "single-output explanations."
            )
        if not np.isfinite(predictions).all():
            raise ValueError("Model predictions must be finite.")
        values[feature_mask] = float(np.dot(global_array, predictions))
        values[feature_mask | geo_bit] = float(np.dot(local_array, predictions))
    return values


def shapley_kernel_weight(n_players: int, coalition_size: int) -> float:
    """Return the finite SHAP kernel weight for an intermediate coalition."""

    if n_players < 2:
        raise ValueError("n_players must be at least two.")
    if coalition_size <= 0 or coalition_size >= n_players:
        raise ValueError("coalition_size must be between one and n_players - 1.")
    return float(
        (n_players - 1)
        / (
            comb(n_players, coalition_size)
            * coalition_size
            * (n_players - coalition_size)
        )
    )


def _geo_design_row(mask: int, n_features: int) -> FloatArray:
    row: FloatArray = np.zeros(2 * n_features + 1, dtype=float)
    geo_bit = 1 << n_features
    geo_present = bool(mask & geo_bit)
    for feature in range(n_features):
        present = bool(mask & (1 << feature))
        if present:
            row[feature] = 1.0
        if present and geo_present:
            row[n_features + 1 + feature] = 1.0
    if geo_present:
        row[n_features] = 1.0
    return row


def exact_geo_decomposition(
    coalition_values: FloatArray,
    n_features: int,
) -> tuple[FloatArray, float, FloatArray, GeoDecompositionDiagnostics]:
    """Decompose an exact joint game into primary, GEO, and GEO interactions.

    Finite-coalition residuals use the SHAP kernel. Full-coalition efficiency is
    imposed as a hard equality constraint rather than an arbitrary large weight.
    """

    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    n_players = n_features + 1
    expected = 1 << n_players
    values: FloatArray = np.asarray(coalition_values, dtype=float)
    if values.shape != (expected,):
        raise ValueError(
            f"coalition_values must have shape ({expected},) for {n_players} players."
        )
    if not np.isfinite(values).all():
        raise ValueError("coalition_values must be finite.")

    masks = np.arange(1, expected - 1, dtype=int)
    design: FloatArray = np.vstack(
        [_geo_design_row(int(mask), n_features) for mask in masks]
    )
    response: FloatArray = values[masks] - values[0]
    weights: FloatArray = np.asarray(
        [
            shapley_kernel_weight(n_players, int(mask).bit_count())
            for mask in masks
        ],
        dtype=float,
    )
    weighted_design: FloatArray = design * np.sqrt(weights)[:, None]
    weighted_response: FloatArray = response * np.sqrt(weights)
    normal: FloatArray = weighted_design.T @ weighted_design
    right: FloatArray = weighted_design.T @ weighted_response

    n_terms = 2 * n_features + 1
    constraint: FloatArray = np.ones(n_terms, dtype=float)
    total_effect = float(values[-1] - values[0])
    kkt: FloatArray = np.zeros((n_terms + 1, n_terms + 1), dtype=float)
    kkt[:n_terms, :n_terms] = normal
    kkt[:n_terms, n_terms] = constraint
    kkt[n_terms, :n_terms] = constraint
    rhs: FloatArray = np.concatenate([right, np.array([total_effect])])

    try:
        solution: FloatArray = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "The exact geographic decomposition system is singular."
        ) from exc

    coefficients: FloatArray = solution[:n_terms]
    primary: FloatArray = np.asarray(coefficients[:n_features], dtype=float)
    geo = float(coefficients[n_features])
    interactions: FloatArray = np.asarray(
        coefficients[n_features + 1 :],
        dtype=float,
    )

    ordinary: FloatArray = exact_shapley_values(values, n_players)
    redistributed: FloatArray = np.concatenate(
        [
            primary + 0.5 * interactions,
            np.array([geo + 0.5 * float(interactions.sum())]),
        ]
    )
    constraint_error = abs(float(constraint @ coefficients) - total_effect)
    equivalence_error = float(np.max(np.abs(redistributed - ordinary)))
    if constraint_error > 1e-8 or equivalence_error > 1e-8:
        raise RuntimeError(
            "The geographic decomposition failed numerical validation."
        )

    diagnostics = GeoDecompositionDiagnostics(
        n_players=n_players,
        n_coalitions=expected,
        design_rank=int(
            np.linalg.matrix_rank(np.vstack([weighted_design, constraint]))
        ),
        condition_number=float(np.linalg.cond(kkt)),
        constraint_error=constraint_error,
        shapley_equivalence_error=equivalence_error,
    )
    return primary, geo, interactions, diagnostics
