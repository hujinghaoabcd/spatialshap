"""Exact explainer for the joint geographic reference-switch game."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from spatialshap._explainer import _as_2d_numeric, _as_geometry, _resolve_predict
from spatialshap._geo_coalitions import (
    GeoDecompositionDiagnostics,
    exact_geo_decomposition,
    exact_joint_coalition_values,
)
from spatialshap._geo_explanation import GeoExplanation
from spatialshap._references import (
    GlobalReference,
    Reference,
    ReferenceDiagnostics,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _infer_feature_names(values: Any) -> tuple[str, ...]:
    if isinstance(values, pd.DataFrame):
        return tuple(str(column) for column in values.columns)
    array = np.asarray(values)
    return tuple(f"feature_{index}" for index in range(array.shape[1]))


class GeoExplainer:
    """Compute an exact four-component joint geographic decomposition.

    The grouped GEO player switches the empirical reference weights from one
    global distribution to the focal location's spatial reference distribution.
    It is not inserted into the predictive model as a fabricated coordinate input.
    """

    def __init__(
        self,
        model: Any,
        background: Any,
        *,
        reference: Reference,
        background_geometry: Any | None = None,
        max_exact_features: int = 11,
    ) -> None:
        self._predict = _resolve_predict(model)
        self._background = _as_2d_numeric(background, name="background")
        self._background_geometry = _as_geometry(
            background_geometry,
            name="background_geometry",
            n_rows=self._background.shape[0],
        )
        if (
            isinstance(max_exact_features, bool)
            or not isinstance(max_exact_features, int)
            or max_exact_features <= 0
        ):
            raise ValueError("max_exact_features must be a positive integer.")
        if self._background.shape[1] > max_exact_features:
            raise ValueError(
                "Exact joint decomposition is disabled above max_exact_features."
            )
        self.max_exact_features = max_exact_features
        self.reference = reference
        self.feature_names = _infer_feature_names(background)
        self._global_weights, self._global_reference_diagnostics = (
            GlobalReference().weights(
                None,
                None,
                self._background.shape[0],
            )
        )
        background_predictions: FloatArray = np.asarray(
            self._predict(self._background),
            dtype=float,
        )
        if background_predictions.shape != (self._background.shape[0],):
            raise ValueError(
                "The model must return one prediction for each background row."
            )
        if not np.isfinite(background_predictions).all():
            raise ValueError("Background predictions must be finite.")
        self.base_value = float(
            np.dot(self._global_weights, background_predictions)
        )

    def __call__(
        self,
        X: Any,
        *,
        geometry: Any | None = None,
    ) -> GeoExplanation:
        data: FloatArray = _as_2d_numeric(X, name="X")
        if data.shape[1] != self._background.shape[1]:
            raise ValueError("X and background must have the same feature count.")
        if isinstance(X, pd.DataFrame):
            names = tuple(str(column) for column in X.columns)
            if names != self.feature_names:
                raise ValueError(
                    "X columns must match the background columns in the same order."
                )
        focal_geometry = _as_geometry(
            geometry,
            name="geometry",
            n_rows=data.shape[0],
        )
        predictions: FloatArray = np.asarray(self._predict(data), dtype=float)
        if predictions.shape != (data.shape[0],):
            raise ValueError("The model must return one prediction for each row.")
        if not np.isfinite(predictions).all():
            raise ValueError("Predictions must be finite.")

        primary: FloatArray = np.empty_like(data, dtype=float)
        geo: FloatArray = np.empty(data.shape[0], dtype=float)
        interactions: FloatArray = np.empty_like(data, dtype=float)
        base_values: FloatArray = np.full(
            data.shape[0],
            self.base_value,
            dtype=float,
        )
        reference_diagnostics: list[ReferenceDiagnostics] = []
        decomposition_diagnostics: list[GeoDecompositionDiagnostics] = []

        for row in range(data.shape[0]):
            focal = None if focal_geometry is None else focal_geometry[row]
            local_weights, reference_diagnostic = self.reference.weights(
                focal,
                self._background_geometry,
                self._background.shape[0],
            )
            coalition_values: FloatArray = exact_joint_coalition_values(
                self._predict,
                data[row],
                self._background,
                self._global_weights,
                local_weights,
            )
            row_primary, row_geo, row_interactions, row_diagnostic = (
                exact_geo_decomposition(
                    coalition_values,
                    n_features=data.shape[1],
                )
            )
            primary[row] = row_primary
            geo[row] = row_geo
            interactions[row] = row_interactions
            reference_diagnostics.append(reference_diagnostic)
            decomposition_diagnostics.append(row_diagnostic)

        result = GeoExplanation(
            primary_values=primary,
            geo_values=geo,
            interaction_values=interactions,
            base_values=base_values,
            data=data,
            predictions=predictions,
            feature_names=self.feature_names,
            geometry=focal_geometry,
            reference_diagnostics=tuple(reference_diagnostics),
            decomposition_diagnostics=tuple(decomposition_diagnostics),
            metadata={
                "algorithm": "exact_joint_geo_kernel",
                "geo_player": "reference_switch",
                "global_reference": "GlobalReference",
                "local_reference": type(self.reference).__name__,
                "n_background": self._background.shape[0],
                "n_features": self._background.shape[1],
            },
        )
        if not np.allclose(result.base_values, self.base_value):
            raise RuntimeError("The joint decomposition base value is not global.")
        return result


def explain_geo(
    model: Any,
    X: Any,
    *,
    background: Any,
    reference: Reference,
    geometry: Any | None = None,
    background_geometry: Any | None = None,
    max_exact_features: int = 11,
) -> GeoExplanation:
    """Explain ``X`` using the exact joint geographic reference-switch game."""

    return GeoExplainer(
        model,
        background,
        reference=reference,
        background_geometry=background_geometry,
        max_exact_features=max_exact_features,
    )(X, geometry=geometry)
