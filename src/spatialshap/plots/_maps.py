"""Coordinate-based SpatialSHAP maps."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from spatialshap._explanation import FloatArray, SpatialExplanation
from spatialshap.plots._summary import _plt


def _feature_index(explanation: SpatialExplanation, feature: str | int) -> int:
    if isinstance(feature, str):
        try:
            return explanation.feature_names.index(feature)
        except ValueError as exc:
            raise KeyError(feature) from exc
    index = int(feature)
    if index < 0 or index >= explanation.shape[1]:
        raise IndexError(index)
    return index


def _require_geometry(explanation: SpatialExplanation) -> FloatArray:
    if explanation.geometry is None:
        raise ValueError("This plot requires geometry coordinates.")
    return cast(FloatArray, explanation.geometry)


def effect_map(
    explanation: SpatialExplanation,
    *,
    feature: str | int,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Map one feature's spatially conditioned contribution."""

    plt = _plt()
    geometry = _require_geometry(explanation)
    index = _feature_index(explanation, feature)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    artist = ax.scatter(
        geometry[:, 0],
        geometry[:, 1],
        c=explanation.values[:, index],
        cmap="coolwarm",
        s=36,
    )
    fig.colorbar(artist, ax=ax, label="Contribution")
    ax.set_title(f"Spatial contribution: {explanation.feature_names[index]}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax


def baseline_map(
    explanation: SpatialExplanation,
    *,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Map the location-specific reference prediction."""

    plt = _plt()
    geometry = _require_geometry(explanation)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    artist = ax.scatter(
        geometry[:, 0],
        geometry[:, 1],
        c=explanation.base_values,
        cmap="viridis",
        s=36,
    )
    fig.colorbar(artist, ax=ax, label="Local reference prediction")
    ax.set_title("SpatialSHAP local baseline")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax


def effective_reference_map(
    explanation: SpatialExplanation,
    *,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Map effective local background sample size."""

    plt = _plt()
    geometry = _require_geometry(explanation)
    effective = np.array(
        [item.effective_n for item in explanation.reference_diagnostics],
        dtype=float,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    artist = ax.scatter(
        geometry[:, 0],
        geometry[:, 1],
        c=effective,
        cmap="viridis",
        s=36,
    )
    fig.colorbar(artist, ax=ax, label="Effective reference sample size")
    ax.set_title("SpatialSHAP reference support")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax
