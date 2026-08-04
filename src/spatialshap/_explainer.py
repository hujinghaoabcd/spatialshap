"""Public exact explainer for spatially conditioned Shapley values."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from spatialshap._coalitions import exact_coalition_values, exact_shapley_values
from spatialshap._explanation import SpatialExplanation
from spatialshap._references import GlobalReference, Reference, ReferenceDiagnostics

FloatArray = NDArray[np.float64]
PredictionFunction = Callable[[FloatArray], FloatArray]


def _as_2d_numeric(values: Any, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional numeric array.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one feature.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.asarray(array, dtype=float)


def _as_geometry(values: Any | None, *, name: str, n_rows: int) -> FloatArray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.shape != (n_rows, 2):
        raise ValueError(f"{name} must have shape ({n_rows}, 2).")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    return array


def _resolve_predict(model: Any) -> PredictionFunction:
    if callable(model):
        return model
    predict = getattr(model, "predict", None)
    if callable(predict):
        return predict
    raise TypeError("model must be callable or expose a callable predict method.")


class Explainer:
    """Compute exact spatially conditioned Shapley values.

    The first implementation deliberately supports only single-output predictions
    and exact coalition enumeration. It serves as a numerical reference for later
    approximation and tree-specific algorithms.
    """

    def __init__(
        self,
        model: Any,
        background: Any,
        *,
        background_geometry: Any | None = None,
        reference: Reference | None = None,
        max_exact_features: int = 12,
    ) -> None:
        self._predict = _resolve_predict(model)
        self._background = _as_2d_numeric(background, name="background")
        self._background_geometry = _as_geometry(
            background_geometry,
            name="background_geometry",
            n_rows=self._background.shape[0],
        )
        self.reference: Reference = reference or GlobalReference()
        if (
            isinstance(max_exact_features, bool)
            or not isinstance(max_exact_features, int)
            or max_exact_features <= 0
        ):
            raise ValueError("max_exact_features must be a positive integer.")
        self.max_exact_features = max_exact_features
        if self._background.shape[1] > self.max_exact_features:
            raise ValueError(
                "Exact enumeration is disabled above max_exact_features; reduce "
                "the feature set or increase the explicit safety limit."
            )
        self.feature_names = self._infer_feature_names(background)
        self._validate_background_predictions()

    @staticmethod
    def _infer_feature_names(background: Any) -> tuple[str, ...]:
        if isinstance(background, pd.DataFrame):
            return tuple(str(column) for column in background.columns)
        array = np.asarray(background)
        return tuple(f"feature_{index}" for index in range(array.shape[1]))

    def _validate_background_predictions(self) -> None:
        prediction = np.asarray(self._predict(self._background), dtype=float)
        if prediction.shape != (self._background.shape[0],):
            raise ValueError(
                "The model must return a one-dimensional prediction for each row."
            )
        if not np.isfinite(prediction).all():
            raise ValueError("Background predictions must be finite.")

    def __call__(
        self,
        X: Any,
        *,
        geometry: Any | None = None,
    ) -> SpatialExplanation:
        data = _as_2d_numeric(X, name="X")
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

        values = np.empty_like(data, dtype=float)
        base_values = np.empty(data.shape[0], dtype=float)
        predictions = np.asarray(self._predict(data), dtype=float)
        if predictions.shape != (data.shape[0],):
            raise ValueError(
                "The model must return a one-dimensional prediction for each row."
            )
        if not np.isfinite(predictions).all():
            raise ValueError("Predictions must be finite.")

        diagnostics: list[ReferenceDiagnostics] = []
        for row in range(data.shape[0]):
            focal = None if focal_geometry is None else focal_geometry[row]
            weights, diagnostic = self.reference.weights(
                focal,
                self._background_geometry,
                self._background.shape[0],
            )
            coalition_values = exact_coalition_values(
                self._predict,
                data[row],
                self._background,
                weights,
            )
            base_values[row] = coalition_values[0]
            values[row] = exact_shapley_values(
                coalition_values,
                n_features=data.shape[1],
            )
            diagnostics.append(diagnostic)

        return SpatialExplanation(
            values=values,
            base_values=base_values,
            data=data,
            predictions=predictions,
            feature_names=self.feature_names,
            geometry=focal_geometry,
            reference_diagnostics=tuple(diagnostics),
            metadata={
                "algorithm": "exact",
                "reference": type(self.reference).__name__,
                "n_background": self._background.shape[0],
                "n_features": self._background.shape[1],
            },
        )


def explain(
    model: Any,
    X: Any,
    *,
    background: Any,
    geometry: Any | None = None,
    background_geometry: Any | None = None,
    reference: Reference | None = None,
    max_exact_features: int = 12,
) -> SpatialExplanation:
    """Explain ``X`` using a one-shot exact SpatialSHAP workflow."""

    return Explainer(
        model,
        background,
        background_geometry=background_geometry,
        reference=reference,
        max_exact_features=max_exact_features,
    )(X, geometry=geometry)
