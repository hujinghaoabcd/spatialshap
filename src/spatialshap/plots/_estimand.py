"""Plots comparing coordinate-group and reference-switch geographic games."""

from __future__ import annotations

from typing import Any

import numpy as np

from spatialshap.estimand_comparison import (
    FloatArray,
    GeographicEstimandComparison,
)
from spatialshap.plots._summary import _plt


def _feature_index(
    comparison: GeographicEstimandComparison,
    feature: str | int | None,
) -> int:
    if feature is None:
        raise ValueError("feature is required for this comparison component.")
    if isinstance(feature, str):
        try:
            return comparison.feature_names.index(feature)
        except ValueError as exc:
            raise KeyError(feature) from exc
    index = int(feature)
    if index < 0 or index >= comparison.shape[1]:
        raise IndexError(index)
    return index


def _component_values(
    comparison: GeographicEstimandComparison,
    *,
    component: str,
    feature: str | int | None,
) -> tuple[FloatArray, FloatArray, str]:
    coordinate = comparison.coordinate_group
    reference = comparison.reference_switch
    if component == "baseline":
        return coordinate.base_values, reference.base_values, "Baseline"
    if component == "geo_main":
        return coordinate.geo_values, reference.geo_values, "GEO main"
    if component == "joint_geo_shapley":
        return (
            coordinate.shapley_values[:, -1],
            reference.shapley_values[:, -1],
            "Joint GEO Shapley",
        )
    index = _feature_index(comparison, feature)
    name = comparison.feature_names[index]
    if component == "primary":
        return (
            coordinate.primary_values[:, index],
            reference.primary_values[:, index],
            f"Primary: {name}",
        )
    if component == "geo_interaction":
        return (
            coordinate.interaction_values[:, index],
            reference.interaction_values[:, index],
            f"GEO interaction: {name}",
        )
    if component == "joint_feature_shapley":
        return (
            coordinate.shapley_values[:, index],
            reference.shapley_values[:, index],
            f"Joint feature Shapley: {name}",
        )
    raise KeyError(component)


def estimand_scatter(
    comparison: GeographicEstimandComparison,
    *,
    component: str = "geo_main",
    feature: str | int | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot coordinate-group values against reference-switch values."""

    coordinate, reference, label = _component_values(
        comparison,
        component=component,
        feature=feature,
    )
    if not np.isfinite(coordinate).all() or not np.isfinite(reference).all():
        raise RuntimeError("Estimand comparison values must be finite.")
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    ax.scatter(coordinate, reference, s=36)
    lower = float(min(np.min(coordinate), np.min(reference)))
    upper = float(max(np.max(coordinate), np.max(reference)))
    if np.isclose(lower, upper):
        margin = max(abs(lower), 1.0) * 0.05
        lower -= margin
        upper += margin
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Coordinate-group game")
    ax.set_ylabel("Reference-switch game")
    ax.set_title(f"Geographic estimand comparison — {label}")
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def estimand_difference_map(
    comparison: GeographicEstimandComparison,
    *,
    component: str = "geo_main",
    feature: str | int | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Map reference-switch minus coordinate-group component values."""

    coordinate, reference, label = _component_values(
        comparison,
        component=component,
        feature=feature,
    )
    difference: FloatArray = np.asarray(reference - coordinate, dtype=float)
    if not np.isfinite(difference).all():
        raise RuntimeError("Estimand comparison differences must be finite.")
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    artist = ax.scatter(
        comparison.geometry[:, 0],
        comparison.geometry[:, 1],
        c=difference,
        cmap="coolwarm",
        s=36,
    )
    fig.colorbar(
        artist,
        ax=ax,
        label="Reference-switch minus coordinate-group",
    )
    ax.set_title(f"Geographic estimand difference — {label}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax
