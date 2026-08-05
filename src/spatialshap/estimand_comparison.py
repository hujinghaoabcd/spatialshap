"""Exact comparison of coordinate-group and reference-switch geographic games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from spatialshap._explainer import _as_2d_numeric, _as_geometry, _resolve_predict
from spatialshap._geo_coalitions import (
    GeoDecompositionDiagnostics,
    exact_geo_decomposition,
)
from spatialshap._geo_explainer import _infer_feature_names
from spatialshap._references import GlobalReference, Reference, ReferenceDiagnostics

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


def _normalized_weights(values: Any, *, n_rows: int, name: str) -> FloatArray:
    weights: FloatArray = np.asarray(values, dtype=float)
    if weights.shape != (n_rows,):
        raise ValueError(f"{name} must have shape ({n_rows},).")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"{name} must contain positive total weight.")
    return np.asarray(weights / total, dtype=float)


def _joint_predictions(
    predict: Any,
    features: FloatArray,
    geometry: FloatArray,
) -> FloatArray:
    joint: FloatArray = np.column_stack([features, geometry])
    predictions: FloatArray = np.asarray(predict(joint), dtype=float)
    if predictions.shape != (features.shape[0],):
        raise ValueError(
            "The joint model must return shape (n_samples,) for feature and "
            "coordinate inputs."
        )
    if not np.isfinite(predictions).all():
        raise ValueError("Joint model predictions must be finite.")
    return predictions


def exact_coordinate_group_coalition_values(
    predict_joint: Any,
    target: Any,
    target_geometry: Any,
    background: Any,
    background_geometry: Any,
    global_weights: Any,
) -> FloatArray:
    """Evaluate the exact grouped-coordinate game used by Kernel GeoShapley.

    Non-geographic features are ordinary players. The last grouped player switches
    all coordinate columns from each background row to the focal coordinates.
    Background rows and their coordinates remain paired when the location player
    is absent.
    """

    predict = _resolve_predict(predict_joint)
    target_array = _readonly_float_array(target, ndim=1, name="target")
    background_array = _readonly_float_array(
        background,
        ndim=2,
        name="background",
    )
    target_location = _readonly_float_array(
        target_geometry,
        ndim=1,
        name="target_geometry",
    )
    background_locations = _readonly_float_array(
        background_geometry,
        ndim=2,
        name="background_geometry",
    )
    if background_array.shape[1] != target_array.size:
        raise ValueError("target and background must share a feature count.")
    if background_locations.shape[0] != background_array.shape[0]:
        raise ValueError("background and background_geometry must share rows.")
    if background_locations.shape[1] != target_location.size:
        raise ValueError(
            "target_geometry and background_geometry must share a coordinate count."
        )
    weights = _normalized_weights(
        global_weights,
        n_rows=background_array.shape[0],
        name="global_weights",
    )

    n_features = int(target_array.size)
    geo_bit = 1 << n_features
    values: FloatArray = np.empty(1 << (n_features + 1), dtype=float)
    focal_locations: FloatArray = np.broadcast_to(
        target_location,
        background_locations.shape,
    )
    for feature_mask in range(1 << n_features):
        mixed = background_array.copy()
        for feature in range(n_features):
            if feature_mask & (1 << feature):
                mixed[:, feature] = target_array[feature]
        global_location_predictions = _joint_predictions(
            predict,
            mixed,
            background_locations,
        )
        focal_location_predictions = _joint_predictions(
            predict,
            mixed,
            focal_locations,
        )
        values[feature_mask] = float(
            np.dot(weights, global_location_predictions)
        )
        values[feature_mask | geo_bit] = float(
            np.dot(weights, focal_location_predictions)
        )
    return values


def exact_reference_switch_joint_coalition_values(
    predict_joint: Any,
    target: Any,
    target_geometry: Any,
    background: Any,
    global_weights: Any,
    local_weights: Any,
) -> FloatArray:
    """Evaluate a location-aware reference-switch game.

    Focal coordinates are fixed in every coalition. The grouped GEO player changes
    only the empirical weights used to integrate absent non-geographic features.
    """

    predict = _resolve_predict(predict_joint)
    target_array = _readonly_float_array(target, ndim=1, name="target")
    background_array = _readonly_float_array(
        background,
        ndim=2,
        name="background",
    )
    target_location = _readonly_float_array(
        target_geometry,
        ndim=1,
        name="target_geometry",
    )
    if background_array.shape[1] != target_array.size:
        raise ValueError("target and background must share a feature count.")
    n_background = int(background_array.shape[0])
    global_array = _normalized_weights(
        global_weights,
        n_rows=n_background,
        name="global_weights",
    )
    local_array = _normalized_weights(
        local_weights,
        n_rows=n_background,
        name="local_weights",
    )

    n_features = int(target_array.size)
    geo_bit = 1 << n_features
    values: FloatArray = np.empty(1 << (n_features + 1), dtype=float)
    focal_locations: FloatArray = np.broadcast_to(
        target_location,
        (n_background, target_location.size),
    )
    for feature_mask in range(1 << n_features):
        mixed = background_array.copy()
        for feature in range(n_features):
            if feature_mask & (1 << feature):
                mixed[:, feature] = target_array[feature]
        predictions = _joint_predictions(predict, mixed, focal_locations)
        values[feature_mask] = float(np.dot(global_array, predictions))
        values[feature_mask | geo_bit] = float(
            np.dot(local_array, predictions)
        )
    return values


@dataclass(frozen=True)
class GeographicGameComponents:
    """Exact components for one geographic cooperative-game contract."""

    contract: str
    primary_values: FloatArray
    geo_values: FloatArray
    interaction_values: FloatArray
    base_values: FloatArray
    predictions: FloatArray
    feature_names: tuple[str, ...]
    decomposition_diagnostics: tuple[GeoDecompositionDiagnostics, ...]

    def __post_init__(self) -> None:
        if not self.contract:
            raise ValueError("contract must not be empty.")
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
        n_rows, n_features = primary.shape
        if interaction.shape != primary.shape:
            raise ValueError("interaction_values must match primary_values.")
        if geo.shape != (n_rows,) or base.shape != (n_rows,):
            raise ValueError("geo_values and base_values must match rows.")
        if predictions.shape != (n_rows,):
            raise ValueError("predictions must match rows.")
        if len(self.feature_names) != n_features:
            raise ValueError("feature_names must match columns.")
        if len(self.decomposition_diagnostics) != n_rows:
            raise ValueError("decomposition_diagnostics must match rows.")
        object.__setattr__(self, "primary_values", primary)
        object.__setattr__(self, "interaction_values", interaction)
        object.__setattr__(self, "geo_values", geo)
        object.__setattr__(self, "base_values", base)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(
            self,
            "decomposition_diagnostics",
            tuple(self.decomposition_diagnostics),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(n_observations, n_non_geographic_features)``."""

        return int(self.primary_values.shape[0]), int(self.primary_values.shape[1])

    @property
    def shapley_values(self) -> FloatArray:
        """Return ordinary feature and grouped-GEO Shapley values."""

        feature = self.primary_values + 0.5 * self.interaction_values
        geo = self.geo_values + 0.5 * self.interaction_values.sum(axis=1)
        values: FloatArray = np.column_stack([feature, geo])
        values.setflags(write=False)
        return values

    @property
    def additivity_error(self) -> FloatArray:
        """Return prediction minus the four-component reconstruction."""

        reconstructed = (
            self.base_values
            + self.primary_values.sum(axis=1)
            + self.geo_values
            + self.interaction_values.sum(axis=1)
        )
        error: FloatArray = self.predictions - reconstructed
        error.setflags(write=False)
        return error


@dataclass(frozen=True)
class GeographicEstimandComparison:
    """Matched comparison of coordinate-group and reference-switch games."""

    coordinate_group: GeographicGameComponents
    reference_switch: GeographicGameComponents
    data: FloatArray
    geometry: FloatArray
    reference_diagnostics: tuple[ReferenceDiagnostics, ...]

    def __post_init__(self) -> None:
        if self.coordinate_group.contract != "coordinate_group":
            raise ValueError("coordinate_group has the wrong contract label.")
        if self.reference_switch.contract != "reference_switch":
            raise ValueError("reference_switch has the wrong contract label.")
        if self.coordinate_group.shape != self.reference_switch.shape:
            raise ValueError("The compared games must have the same shape.")
        if self.coordinate_group.feature_names != self.reference_switch.feature_names:
            raise ValueError("The compared games must share feature names.")
        data = _readonly_float_array(self.data, ndim=2, name="data")
        geometry = _readonly_float_array(self.geometry, ndim=2, name="geometry")
        if data.shape != self.coordinate_group.shape:
            raise ValueError("data must match the comparison shape.")
        if geometry.shape[0] != data.shape[0]:
            raise ValueError("geometry must match the observation count.")
        if len(self.reference_diagnostics) != data.shape[0]:
            raise ValueError("reference_diagnostics must match observations.")
        if not np.allclose(
            self.coordinate_group.predictions,
            self.reference_switch.predictions,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("The compared games must reconstruct the same prediction.")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(
            self,
            "reference_diagnostics",
            tuple(self.reference_diagnostics),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shared observation and feature shape."""

        return self.coordinate_group.shape

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Return shared non-geographic feature names."""

        return self.coordinate_group.feature_names

    def component_table(self) -> pd.DataFrame:
        """Summarize agreement and disagreement between the two estimands."""

        coordinate = self.coordinate_group
        reference = self.reference_switch
        comparisons = {
            "baseline": (coordinate.base_values, reference.base_values),
            "primary": (coordinate.primary_values, reference.primary_values),
            "geo_main": (coordinate.geo_values, reference.geo_values),
            "geo_interaction": (
                coordinate.interaction_values,
                reference.interaction_values,
            ),
            "joint_feature_shapley": (
                coordinate.shapley_values[:, :-1],
                reference.shapley_values[:, :-1],
            ),
            "joint_geo_shapley": (
                coordinate.shapley_values[:, -1],
                reference.shapley_values[:, -1],
            ),
        }
        rows: list[dict[str, float | str]] = []
        for component, (coordinate_values, reference_values) in comparisons.items():
            first = np.asarray(coordinate_values, dtype=float).reshape(-1)
            second = np.asarray(reference_values, dtype=float).reshape(-1)
            difference = second - first
            first_std = float(np.std(first))
            second_std = float(np.std(second))
            if first_std > 0 and second_std > 0:
                correlation = float(np.corrcoef(first, second)[0, 1])
            elif np.allclose(first, second):
                correlation = 1.0
            else:
                correlation = float("nan")
            rows.append(
                {
                    "component": component,
                    "mean_abs_coordinate_group": float(np.mean(np.abs(first))),
                    "mean_abs_reference_switch": float(np.mean(np.abs(second))),
                    "mean_abs_difference": float(np.mean(np.abs(difference))),
                    "rmse_difference": float(
                        np.sqrt(np.mean(np.square(difference)))
                    ),
                    "correlation": correlation,
                }
            )
        return pd.DataFrame(rows).set_index("component")

    def to_frame(self) -> pd.DataFrame:
        """Return observation-level components under both game contracts."""

        coordinate = self.coordinate_group
        reference = self.reference_switch
        frame = pd.DataFrame(
            {
                "prediction": coordinate.predictions,
                "coordinate_group__base": coordinate.base_values,
                "reference_switch__base": reference.base_values,
                "difference__base": reference.base_values - coordinate.base_values,
                "coordinate_group__geo_main": coordinate.geo_values,
                "reference_switch__geo_main": reference.geo_values,
                "difference__geo_main": reference.geo_values - coordinate.geo_values,
                "coordinate_group__joint_geo_shapley": coordinate.shapley_values[:, -1],
                "reference_switch__joint_geo_shapley": reference.shapley_values[:, -1],
                "difference__joint_geo_shapley": (
                    reference.shapley_values[:, -1]
                    - coordinate.shapley_values[:, -1]
                ),
                "coordinate_group__additivity_error": coordinate.additivity_error,
                "reference_switch__additivity_error": reference.additivity_error,
            }
        )
        for index, name in enumerate(self.feature_names):
            frame[f"value__{name}"] = self.data[:, index]
            for component, coordinate_values, reference_values in (
                (
                    "primary",
                    coordinate.primary_values[:, index],
                    reference.primary_values[:, index],
                ),
                (
                    "geo_interaction",
                    coordinate.interaction_values[:, index],
                    reference.interaction_values[:, index],
                ),
                (
                    "joint_shapley",
                    coordinate.shapley_values[:, index],
                    reference.shapley_values[:, index],
                ),
            ):
                frame[f"coordinate_group__{component}__{name}"] = coordinate_values
                frame[f"reference_switch__{component}__{name}"] = reference_values
                frame[f"difference__{component}__{name}"] = (
                    reference_values - coordinate_values
                )
        frame["geometry_x"] = self.geometry[:, 0]
        frame["geometry_y"] = self.geometry[:, 1]
        return frame

    def summary(self) -> str:
        """Return a compact scientific comparison summary."""

        table = self.component_table()
        maximum_additivity = max(
            float(np.max(np.abs(self.coordinate_group.additivity_error))),
            float(np.max(np.abs(self.reference_switch.additivity_error))),
        )
        return (
            "GeographicEstimandComparison("
            f"n={self.shape[0]}, p={self.shape[1]}, "
            f"max_additivity_error={maximum_additivity:.3e})\n"
            "Reference-switch minus coordinate-group differences:\n"
            f"{table.to_string()}"
        )


def compare_geographic_estimands(
    model: Any,
    X: Any,
    *,
    background: Any,
    geometry: Any,
    background_geometry: Any,
    reference: Reference,
    global_weights: Any | None = None,
    max_exact_features: int = 8,
) -> GeographicEstimandComparison:
    """Compare two geographic games for one location-aware prediction model.

    The model receives non-geographic features followed by two coordinate columns.
    The coordinate-group game matches the grouped-location value function used by
    Kernel GeoShapley. The reference-switch game fixes focal coordinates and lets
    GEO switch global empirical weights to the focal spatial reference weights.
    """

    predict = _resolve_predict(model)
    data = _as_2d_numeric(X, name="X")
    background_array = _as_2d_numeric(background, name="background")
    if data.shape[1] != background_array.shape[1]:
        raise ValueError("X and background must share a feature count.")
    if (
        isinstance(max_exact_features, bool)
        or not isinstance(max_exact_features, int)
        or max_exact_features <= 0
    ):
        raise ValueError("max_exact_features must be a positive integer.")
    if data.shape[1] > max_exact_features:
        raise ValueError(
            "Exact estimand comparison is disabled above max_exact_features."
        )
    feature_names = _infer_feature_names(background)
    if isinstance(X, pd.DataFrame):
        names = tuple(str(column) for column in X.columns)
        if names != feature_names:
            raise ValueError(
                "X columns must match background columns in the same order."
            )
    focal_geometry = _as_geometry(
        geometry,
        name="geometry",
        n_rows=data.shape[0],
    )
    background_locations = _as_geometry(
        background_geometry,
        name="background_geometry",
        n_rows=background_array.shape[0],
    )
    if focal_geometry is None or background_locations is None:
        raise ValueError("geometry and background_geometry are required.")

    if global_weights is None:
        global_array, _ = GlobalReference().weights(
            None,
            None,
            background_array.shape[0],
        )
    else:
        global_array = _normalized_weights(
            global_weights,
            n_rows=background_array.shape[0],
            name="global_weights",
        )

    target_joint: FloatArray = np.column_stack([data, focal_geometry])
    predictions: FloatArray = np.asarray(predict(target_joint), dtype=float)
    if predictions.shape != (data.shape[0],):
        raise ValueError("The joint model must return one prediction per X row.")
    if not np.isfinite(predictions).all():
        raise ValueError("Target predictions must be finite.")

    coordinate_primary: FloatArray = np.empty_like(data, dtype=float)
    coordinate_geo: FloatArray = np.empty(data.shape[0], dtype=float)
    coordinate_interaction: FloatArray = np.empty_like(data, dtype=float)
    coordinate_base: FloatArray = np.empty(data.shape[0], dtype=float)
    reference_primary: FloatArray = np.empty_like(data, dtype=float)
    reference_geo: FloatArray = np.empty(data.shape[0], dtype=float)
    reference_interaction: FloatArray = np.empty_like(data, dtype=float)
    reference_base: FloatArray = np.empty(data.shape[0], dtype=float)
    coordinate_diagnostics: list[GeoDecompositionDiagnostics] = []
    switch_diagnostics: list[GeoDecompositionDiagnostics] = []
    reference_diagnostics: list[ReferenceDiagnostics] = []

    for row in range(data.shape[0]):
        local_weights, reference_diagnostic = reference.weights(
            focal_geometry[row],
            background_locations,
            background_array.shape[0],
        )
        coordinate_values = exact_coordinate_group_coalition_values(
            predict,
            data[row],
            focal_geometry[row],
            background_array,
            background_locations,
            global_array,
        )
        switch_values = exact_reference_switch_joint_coalition_values(
            predict,
            data[row],
            focal_geometry[row],
            background_array,
            global_array,
            local_weights,
        )
        (
            coordinate_primary[row],
            coordinate_geo[row],
            coordinate_interaction[row],
            coordinate_diagnostic,
        ) = exact_geo_decomposition(coordinate_values, data.shape[1])
        (
            reference_primary[row],
            reference_geo[row],
            reference_interaction[row],
            switch_diagnostic,
        ) = exact_geo_decomposition(switch_values, data.shape[1])
        coordinate_base[row] = coordinate_values[0]
        reference_base[row] = switch_values[0]
        if not np.allclose(
            [coordinate_values[-1], switch_values[-1]],
            predictions[row],
            atol=1e-10,
            rtol=1e-10,
        ):
            raise RuntimeError(
                "The compared games do not share the target-model prediction."
            )
        coordinate_diagnostics.append(coordinate_diagnostic)
        switch_diagnostics.append(switch_diagnostic)
        reference_diagnostics.append(reference_diagnostic)

    return GeographicEstimandComparison(
        coordinate_group=GeographicGameComponents(
            contract="coordinate_group",
            primary_values=coordinate_primary,
            geo_values=coordinate_geo,
            interaction_values=coordinate_interaction,
            base_values=coordinate_base,
            predictions=predictions,
            feature_names=feature_names,
            decomposition_diagnostics=tuple(coordinate_diagnostics),
        ),
        reference_switch=GeographicGameComponents(
            contract="reference_switch",
            primary_values=reference_primary,
            geo_values=reference_geo,
            interaction_values=reference_interaction,
            base_values=reference_base,
            predictions=predictions,
            feature_names=feature_names,
            decomposition_diagnostics=tuple(switch_diagnostics),
        ),
        data=data,
        geometry=focal_geometry,
        reference_diagnostics=tuple(reference_diagnostics),
    )
