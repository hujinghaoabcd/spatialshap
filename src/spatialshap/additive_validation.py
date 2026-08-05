"""Analytic truth for nonlinear additive reference-switch games."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from spatialshap.validation import (
    FloatArray,
    LinearReferenceSwitchTruth,
    _normalized_weights,
    _readonly_float_array,
)


class AdditiveReferenceSwitchTruth(LinearReferenceSwitchTruth):
    """Analytic truth for an additive, potentially nonlinear prediction function."""


def _evaluate_feature_functions(
    data: FloatArray,
    feature_functions: Sequence[Callable[[FloatArray], Any]],
    *,
    name: str,
) -> FloatArray:
    n_rows, n_features = data.shape
    if len(feature_functions) != n_features:
        raise ValueError("feature_functions must match the feature count.")
    effects: FloatArray = np.empty((n_rows, n_features), dtype=float)
    for feature, function in enumerate(feature_functions):
        if not callable(function):
            raise TypeError("Every feature function must be callable.")
        values: FloatArray = np.asarray(function(data[:, feature]), dtype=float)
        if values.shape != (n_rows,):
            raise ValueError(
                f"Feature function {feature} must return shape ({n_rows},) for {name}."
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Feature function {feature} returned non-finite values for {name}."
            )
        effects[:, feature] = values
    return effects


def additive_reference_switch_truth(
    X: Any,
    feature_functions: Sequence[Callable[[FloatArray], Any]],
    *,
    background: Any,
    local_weights: Any,
    intercept: float = 0.0,
    global_weights: Any | None = None,
) -> AdditiveReferenceSwitchTruth:
    """Return exact truth for ``intercept + sum_j g_j(X_j)``.

    Each feature function is evaluated independently on the focal and background
    columns. This permits smooth nonlinear, threshold, and piecewise additive
    responses while retaining an exact four-component reference-switch game.
    """

    data = _readonly_float_array(X, ndim=2, name="X")
    reference = _readonly_float_array(background, ndim=2, name="background")
    if data.shape[1] != reference.shape[1]:
        raise ValueError("X and background must share a feature count.")
    if not np.isfinite(intercept):
        raise ValueError("intercept must be finite.")

    target_effects = _evaluate_feature_functions(
        data,
        feature_functions,
        name="X",
    )
    background_effects = _evaluate_feature_functions(
        reference,
        feature_functions,
        name="background",
    )
    n_rows = int(data.shape[0])
    n_background = int(reference.shape[0])
    local = _normalized_weights(
        local_weights,
        shape=(n_rows, n_background),
        name="local_weights",
    )
    if global_weights is None:
        global_weight: FloatArray = np.full(
            n_background,
            1.0 / n_background,
            dtype=float,
        )
    else:
        global_weight = _normalized_weights(
            global_weights,
            shape=(n_background,),
            name="global_weights",
        )

    global_effect_mean: FloatArray = global_weight @ background_effects
    local_effect_means: FloatArray = local @ background_effects
    primary: FloatArray = target_effects - global_effect_mean
    shifts: FloatArray = local_effect_means - global_effect_mean
    geo: FloatArray = shifts.sum(axis=1)
    interactions: FloatArray = -shifts
    base_value = float(intercept + global_effect_mean.sum())
    base: FloatArray = np.full(n_rows, base_value, dtype=float)
    predictions: FloatArray = intercept + target_effects.sum(axis=1)
    return AdditiveReferenceSwitchTruth(
        primary_values=primary,
        geo_values=geo,
        interaction_values=interactions,
        base_values=base,
        predictions=predictions,
    )
