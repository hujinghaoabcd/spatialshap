"""Exact coalition evaluation and Shapley allocation."""

from __future__ import annotations

from collections.abc import Callable
from math import factorial

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def shapley_weight(n_features: int, coalition_size: int) -> float:
    """Return the exact Shapley permutation weight for one coalition size."""

    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    if coalition_size < 0 or coalition_size >= n_features:
        raise ValueError("coalition_size must be in [0, n_features - 1].")
    return float(
        factorial(coalition_size)
        * factorial(n_features - coalition_size - 1)
        / factorial(n_features)
    )


def exact_coalition_values(
    predict: Callable[[FloatArray], FloatArray],
    target: FloatArray,
    background: FloatArray,
    reference_weights: FloatArray,
) -> FloatArray:
    """Evaluate every coalition under a weighted empirical reference measure."""

    n_features = int(target.shape[0])
    n_coalitions = 1 << n_features
    values = np.empty(n_coalitions, dtype=float)
    for mask in range(n_coalitions):
        mixed = background.copy()
        for feature in range(n_features):
            if mask & (1 << feature):
                mixed[:, feature] = target[feature]
        predictions = np.asarray(predict(mixed), dtype=float)
        if predictions.shape != (background.shape[0],):
            raise ValueError(
                "The prediction callable must return shape (n_samples,) for "
                "single-output explanations."
            )
        if not np.isfinite(predictions).all():
            raise ValueError("Model predictions must be finite.")
        values[mask] = float(np.dot(reference_weights, predictions))
    return values


def exact_shapley_values(coalition_values: FloatArray, n_features: int) -> FloatArray:
    """Allocate exact Shapley values from a full coalition-value vector."""

    expected = 1 << n_features
    if coalition_values.shape != (expected,):
        raise ValueError(
            f"coalition_values must have shape ({expected},) for {n_features} features."
        )
    output = np.zeros(n_features, dtype=float)
    for feature in range(n_features):
        bit = 1 << feature
        for mask in range(expected):
            if mask & bit:
                continue
            coalition_size = mask.bit_count()
            output[feature] += shapley_weight(n_features, coalition_size) * (
                coalition_values[mask | bit] - coalition_values[mask]
            )
    return output
