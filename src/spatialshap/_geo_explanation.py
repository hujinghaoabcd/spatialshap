"""Immutable result object for joint geographic decompositions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from spatialshap._geo_coalitions import GeoDecompositionDiagnostics
from spatialshap._references import ReferenceDiagnostics

FloatArray: TypeAlias = NDArray[np.float64]


def _readonly_float_array(values: Any, *, ndim: int) -> FloatArray:
    array = np.asarray(values, dtype=float).copy()
    if array.ndim != ndim:
        raise ValueError(f"Expected an array with {ndim} dimensions.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GeoExplanation:
    """Four-component geographic Shapley decomposition for observations."""

    primary_values: FloatArray
    geo_values: FloatArray
    interaction_values: FloatArray
    base_values: FloatArray
    data: FloatArray
    predictions: FloatArray
    feature_names: tuple[str, ...]
    geometry: FloatArray | None
    reference_diagnostics: tuple[ReferenceDiagnostics, ...]
    decomposition_diagnostics: tuple[GeoDecompositionDiagnostics, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        primary = _readonly_float_array(self.primary_values, ndim=2)
        geo = _readonly_float_array(self.geo_values, ndim=1)
        interactions = _readonly_float_array(self.interaction_values, ndim=2)
        base = _readonly_float_array(self.base_values, ndim=1)
        data = _readonly_float_array(self.data, ndim=2)
        predictions = _readonly_float_array(self.predictions, ndim=1)
        n_rows, n_features = primary.shape
        if interactions.shape != (n_rows, n_features):
            raise ValueError("interaction_values must match primary_values.")
        if data.shape != (n_rows, n_features):
            raise ValueError("data must match primary_values.")
        if geo.shape != (n_rows,) or base.shape != (n_rows,):
            raise ValueError("geo_values and base_values must match the row count.")
        if predictions.shape != (n_rows,):
            raise ValueError("predictions must match the row count.")
        if len(self.feature_names) != n_features:
            raise ValueError("feature_names must match the feature dimension.")
        if len(self.reference_diagnostics) != n_rows:
            raise ValueError("reference_diagnostics must match the row count.")
        if len(self.decomposition_diagnostics) != n_rows:
            raise ValueError("decomposition_diagnostics must match the row count.")
        geometry = None
        if self.geometry is not None:
            geometry = _readonly_float_array(self.geometry, ndim=2)
            if geometry.shape != (n_rows, 2):
                raise ValueError("geometry must have shape (n_rows, 2).")
        object.__setattr__(self, "primary_values", primary)
        object.__setattr__(self, "geo_values", geo)
        object.__setattr__(self, "interaction_values", interactions)
        object.__setattr__(self, "base_values", base)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(
            self,
            "reference_diagnostics",
            tuple(self.reference_diagnostics),
        )
        object.__setattr__(
            self,
            "decomposition_diagnostics",
            tuple(self.decomposition_diagnostics),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(n_observations, n_non_geographic_features)``."""

        return int(self.primary_values.shape[0]), int(self.primary_values.shape[1])

    @property
    def shapley_feature_names(self) -> tuple[str, ...]:
        """Return feature-player names followed by the grouped GEO player."""

        return self.feature_names + ("GEO",)

    @property
    def shapley_values(self) -> FloatArray:
        """Return ordinary joint-game Shapley values.

        Each GEO–feature interaction is shared equally between the two players.
        """

        feature_values = self.primary_values + 0.5 * self.interaction_values
        geo_values = self.geo_values + 0.5 * self.interaction_values.sum(axis=1)
        values = np.column_stack([feature_values, geo_values])
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
        error = self.predictions - reconstructed
        error.setflags(write=False)
        return error

    @property
    def mean_abs_components(self) -> pd.DataFrame:
        """Return feature-level primary, interaction, and Shapley magnitudes."""

        return pd.DataFrame(
            {
                "mean_abs_primary": np.mean(np.abs(self.primary_values), axis=0),
                "mean_abs_geo_interaction": np.mean(
                    np.abs(self.interaction_values),
                    axis=0,
                ),
                "mean_abs_joint_shapley": np.mean(
                    np.abs(self.shapley_values[:, :-1]),
                    axis=0,
                ),
            },
            index=self.feature_names,
        ).sort_values("mean_abs_joint_shapley", ascending=False)

    def __getitem__(self, key: Any) -> GeoExplanation:
        """Slice rows, or rows and non-geographic features."""

        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("GeoExplanation accepts at most two indices.")
            row_key, feature_key = key
        else:
            row_key, feature_key = key, slice(None)
        row_indices = np.atleast_1d(np.arange(self.shape[0])[row_key])
        if isinstance(feature_key, str):
            try:
                feature_indices = np.array(
                    [self.feature_names.index(feature_key)],
                    dtype=int,
                )
            except ValueError as exc:
                raise KeyError(feature_key) from exc
        else:
            feature_indices = np.atleast_1d(
                np.arange(self.shape[1])[feature_key]
            )
        names = tuple(self.feature_names[int(i)] for i in feature_indices)
        geometry = None if self.geometry is None else self.geometry[row_indices]
        return replace(
            self,
            primary_values=self.primary_values[
                np.ix_(row_indices, feature_indices)
            ],
            geo_values=self.geo_values[row_indices],
            interaction_values=self.interaction_values[
                np.ix_(row_indices, feature_indices)
            ],
            base_values=self.base_values[row_indices],
            data=self.data[np.ix_(row_indices, feature_indices)],
            predictions=self.predictions[row_indices],
            feature_names=names,
            geometry=geometry,
            reference_diagnostics=tuple(
                self.reference_diagnostics[int(i)] for i in row_indices
            ),
            decomposition_diagnostics=tuple(
                self.decomposition_diagnostics[int(i)] for i in row_indices
            ),
        )

    def to_frame(self) -> pd.DataFrame:
        """Return a flat table containing all four components and diagnostics."""

        frame = pd.DataFrame(
            {
                "prediction": self.predictions,
                "base_value": self.base_values,
                "geo_main": self.geo_values,
                "shapley__GEO": self.shapley_values[:, -1],
                "additivity_error": self.additivity_error,
            }
        )
        shapley = self.shapley_values
        for index, name in enumerate(self.feature_names):
            frame[f"value__{name}"] = self.data[:, index]
            frame[f"primary__{name}"] = self.primary_values[:, index]
            frame[f"interaction__GEO__{name}"] = self.interaction_values[:, index]
            frame[f"shapley__{name}"] = shapley[:, index]
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    [item.as_dict() for item in self.reference_diagnostics],
                    index=frame.index,
                ),
                pd.DataFrame(
                    [item.as_dict() for item in self.decomposition_diagnostics],
                    index=frame.index,
                ),
            ],
            axis=1,
        )
        if self.geometry is not None:
            frame["geometry_x"] = self.geometry[:, 0]
            frame["geometry_y"] = self.geometry[:, 1]
        return frame

    def diagnostics(self) -> pd.DataFrame:
        """Return reference, numerical, and structural diagnostics."""

        columns = [
            "additivity_error",
            "reference_effective_n",
            "reference_mean_distance",
            "reference_weight_concentration",
            "decomposition_condition_number",
            "decomposition_weighted_residual_rmse",
            "decomposition_relative_residual_norm",
            "decomposition_max_abs_coalition_residual",
            "decomposition_mean_abs_feature_pair_second_difference",
            "decomposition_max_abs_feature_pair_second_difference",
            "decomposition_constraint_error",
            "decomposition_shapley_equivalence_error",
        ]
        return self.to_frame()[columns]

    def summary(self) -> str:
        """Return a compact terminal-friendly summary."""

        max_additivity = float(np.max(np.abs(self.additivity_error)))
        max_equivalence = max(
            item.shapley_equivalence_error
            for item in self.decomposition_diagnostics
        )
        max_relative_residual = max(
            item.relative_residual_norm
            for item in self.decomposition_diagnostics
        )
        max_feature_pair_difference = max(
            item.max_abs_feature_pair_second_difference
            for item in self.decomposition_diagnostics
        )
        geo_magnitude = float(np.mean(np.abs(self.geo_values)))
        ranked = self.mean_abs_components.to_string()
        return (
            "GeoExplanation("
            f"n={self.shape[0]}, p={self.shape[1]}, "
            f"max_additivity_error={max_additivity:.3e}, "
            f"max_shapley_equivalence_error={max_equivalence:.3e}, "
            f"max_relative_structure_residual={max_relative_residual:.3e}, "
            f"max_feature_pair_second_difference={max_feature_pair_difference:.3e}, "
            f"mean_abs_geo_main={geo_magnitude:.3e})\n"
            f"Feature components:\n{ranked}"
        )

    def to_geodataframe(self, *, crs: str | None = None) -> Any:
        """Return a point GeoDataFrame when GeoPandas is installed."""

        if self.geometry is None:
            raise ValueError("This explanation does not contain geometry.")
        try:
            import geopandas as gpd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "GeoPandas is required; install spatialshap[geo]."
            ) from exc
        frame = self.to_frame().drop(columns=["geometry_x", "geometry_y"])
        points = gpd.points_from_xy(self.geometry[:, 0], self.geometry[:, 1])
        return gpd.GeoDataFrame(frame, geometry=points, crs=crs)
