"""Immutable public result object for SpatialSHAP explanations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from spatialshap._references import ReferenceDiagnostics

FloatArray = NDArray[np.float64]


def _readonly_float_array(values: Any, *, ndim: int) -> FloatArray:
    array = np.asarray(values, dtype=float).copy()
    if array.ndim != ndim:
        raise ValueError(f"Expected an array with {ndim} dimensions.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SpatialExplanation:
    """One immutable set of spatially conditioned feature attributions."""

    values: FloatArray
    base_values: FloatArray
    data: FloatArray
    predictions: FloatArray
    feature_names: tuple[str, ...]
    geometry: FloatArray | None
    reference_diagnostics: tuple[ReferenceDiagnostics, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = _readonly_float_array(self.values, ndim=2)
        base_values = _readonly_float_array(self.base_values, ndim=1)
        data = _readonly_float_array(self.data, ndim=2)
        predictions = _readonly_float_array(self.predictions, ndim=1)
        n_rows, n_features = values.shape
        if data.shape != (n_rows, n_features):
            raise ValueError("data must have the same shape as values.")
        if base_values.shape != (n_rows,) or predictions.shape != (n_rows,):
            raise ValueError("base_values and predictions must match the row count.")
        if len(self.feature_names) != n_features:
            raise ValueError("feature_names must match the feature dimension.")
        if len(self.reference_diagnostics) != n_rows:
            raise ValueError("reference_diagnostics must match the row count.")
        geometry = None
        if self.geometry is not None:
            geometry = _readonly_float_array(self.geometry, ndim=2)
            if geometry.shape != (n_rows, 2):
                raise ValueError("geometry must have shape (n_rows, 2).")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "base_values", base_values)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(
            self,
            "reference_diagnostics",
            tuple(self.reference_diagnostics),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(n_observations, n_features)``."""

        return self.values.shape

    @property
    def additivity_error(self) -> FloatArray:
        """Return prediction minus reconstructed additive explanation."""

        error = self.predictions - (self.base_values + self.values.sum(axis=1))
        error.setflags(write=False)
        return error

    @property
    def mean_abs_values(self) -> pd.Series:
        """Return global mean absolute attribution by feature."""

        return pd.Series(
            np.mean(np.abs(self.values), axis=0),
            index=self.feature_names,
            name="mean_abs_value",
        ).sort_values(ascending=False)

    def __getitem__(self, key: Any) -> SpatialExplanation:
        """Slice rows, or rows and features using ``explanation[rows, columns]``."""

        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("SpatialExplanation accepts at most two indices.")
            row_key, feature_key = key
        else:
            row_key, feature_key = key, slice(None)

        row_indices = np.arange(self.shape[0])[row_key]
        row_indices = np.atleast_1d(row_indices)

        if isinstance(feature_key, str):
            try:
                feature_indices = np.array(
                    [self.feature_names.index(feature_key)], dtype=int
                )
            except ValueError as exc:
                raise KeyError(feature_key) from exc
        else:
            feature_indices = np.arange(self.shape[1])[feature_key]
            feature_indices = np.atleast_1d(feature_indices)

        names = tuple(self.feature_names[int(i)] for i in feature_indices)
        geometry = None if self.geometry is None else self.geometry[row_indices]
        diagnostics = tuple(
            self.reference_diagnostics[int(i)] for i in row_indices
        )
        return replace(
            self,
            values=self.values[np.ix_(row_indices, feature_indices)],
            base_values=self.base_values[row_indices],
            data=self.data[np.ix_(row_indices, feature_indices)],
            predictions=self.predictions[row_indices],
            feature_names=names,
            geometry=geometry,
            reference_diagnostics=diagnostics,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return a flat, auditable table of predictions and contributions."""

        frame = pd.DataFrame(
            {
                "prediction": self.predictions,
                "base_value": self.base_values,
                "additivity_error": self.additivity_error,
            }
        )
        for index, name in enumerate(self.feature_names):
            frame[f"value__{name}"] = self.data[:, index]
            frame[f"phi__{name}"] = self.values[:, index]
        diagnostic_rows = [item.as_dict() for item in self.reference_diagnostics]
        frame = pd.concat(
            [frame, pd.DataFrame(diagnostic_rows, index=frame.index)],
            axis=1,
        )
        if self.geometry is not None:
            frame["geometry_x"] = self.geometry[:, 0]
            frame["geometry_y"] = self.geometry[:, 1]
        return frame

    def to_geodataframe(self, *, crs: str | None = None):
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

    def diagnostics(self) -> pd.DataFrame:
        """Return compact row-level numerical and reference diagnostics."""

        columns = [
            "additivity_error",
            "reference_n_positive",
            "reference_effective_n",
            "reference_mean_distance",
            "reference_max_distance",
            "reference_weight_concentration",
            "reference_fallback_used",
        ]
        return self.to_frame()[columns]

    def summary(self) -> str:
        """Return a compact terminal-friendly explanation summary."""

        max_error = float(np.max(np.abs(self.additivity_error)))
        effective_n = np.array(
            [item.effective_n for item in self.reference_diagnostics],
            dtype=float,
        )
        ranked = self.mean_abs_values.to_string()
        return (
            "SpatialExplanation("
            f"n={self.shape[0]}, p={self.shape[1]}, "
            f"max_additivity_error={max_error:.3e}, "
            f"mean_effective_reference_n={effective_n.mean():.3f})\n"
            f"Mean absolute contributions:\n{ranked}"
        )

    def save(self, path: str) -> None:
        """Save the numerical explanation to a compressed NumPy archive."""

        metadata = dict(self.metadata)
        np.savez_compressed(
            path,
            values=self.values,
            base_values=self.base_values,
            data=self.data,
            predictions=self.predictions,
            feature_names=np.asarray(self.feature_names, dtype=str),
            geometry=(
                np.empty((0, 2), dtype=float)
                if self.geometry is None
                else self.geometry
            ),
            metadata_keys=np.asarray(list(metadata), dtype=str),
            metadata_values=np.asarray([str(v) for v in metadata.values()], dtype=str),
        )
