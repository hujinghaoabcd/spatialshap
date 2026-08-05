"""Plots for the joint geographic decomposition."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from spatialshap._geo_explanation import FloatArray, GeoExplanation
from spatialshap.plots._summary import _plt


def _feature_index(explanation: GeoExplanation, feature: str | int) -> int:
    if isinstance(feature, str):
        try:
            return explanation.feature_names.index(feature)
        except ValueError as exc:
            raise KeyError(feature) from exc
    index = int(feature)
    if index < 0 or index >= explanation.shape[1]:
        raise IndexError(index)
    return index


def _require_geometry(explanation: GeoExplanation) -> FloatArray:
    if explanation.geometry is None:
        raise ValueError("This plot requires geometry coordinates.")
    return cast(FloatArray, explanation.geometry)


def geo_effect_map(
    explanation: GeoExplanation,
    *,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Map the intrinsic GEO main component."""

    plt = _plt()
    geometry = _require_geometry(explanation)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    artist = ax.scatter(
        geometry[:, 0],
        geometry[:, 1],
        c=explanation.geo_values,
        cmap="coolwarm",
        s=36,
    )
    fig.colorbar(artist, ax=ax, label="GEO main contribution")
    ax.set_title("SpatialSHAP intrinsic geographic contribution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax


def interaction_map(
    explanation: GeoExplanation,
    *,
    feature: str | int,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Map one GEO–feature interaction component."""

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
        c=explanation.interaction_values[:, index],
        cmap="coolwarm",
        s=36,
    )
    fig.colorbar(artist, ax=ax, label="GEO–feature interaction")
    ax.set_title(f"GEO interaction: {explanation.feature_names[index]}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax


def structure_diagnostic_map(
    explanation: GeoExplanation,
    *,
    metric: str = "relative_residual_norm",
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Map one coalition-structure diagnostic across focal observations."""

    geometry = _require_geometry(explanation)
    if metric == "weighted_residual_rmse":
        values = np.asarray(
            [item.weighted_residual_rmse for item in explanation.decomposition_diagnostics],
            dtype=float,
        )
        label = "Weighted coalition residual RMSE"
    elif metric == "relative_residual_norm":
        values = np.asarray(
            [item.relative_residual_norm for item in explanation.decomposition_diagnostics],
            dtype=float,
        )
        label = "Relative coalition residual norm"
    elif metric == "max_abs_coalition_residual":
        values = np.asarray(
            [
                item.max_abs_coalition_residual
                for item in explanation.decomposition_diagnostics
            ],
            dtype=float,
        )
        label = "Maximum absolute coalition residual"
    elif metric == "max_abs_feature_pair_second_difference":
        values = np.asarray(
            [
                item.max_abs_feature_pair_second_difference
                for item in explanation.decomposition_diagnostics
            ],
            dtype=float,
        )
        label = "Maximum feature-pair second difference"
    else:
        raise KeyError(metric)

    if not np.isfinite(values).all():
        raise RuntimeError("Structural diagnostic values must be finite.")
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    artist = ax.scatter(
        geometry[:, 0],
        geometry[:, 1],
        c=values,
        cmap="magma",
        s=36,
    )
    fig.colorbar(artist, ax=ax, label=label)
    ax.set_title(f"SpatialSHAP structure diagnostic: {metric}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax


def component_bar(
    explanation: GeoExplanation,
    *,
    max_display: int = 12,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot mean absolute primary and GEO-interaction magnitudes."""

    if max_display <= 0:
        raise ValueError("max_display must be positive.")
    plt = _plt()
    table = explanation.mean_abs_components.iloc[:max_display].iloc[::-1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.42 * len(table) + 1.8)))
    else:
        fig = ax.figure
    y = np.arange(len(table))
    primary = table["mean_abs_primary"].to_numpy()
    interaction = table["mean_abs_geo_interaction"].to_numpy()
    ax.barh(y, primary, label="Primary")
    ax.barh(y, interaction, left=primary, label="GEO interaction")
    ax.set_yticks(y)
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Mean absolute component magnitude")
    mean_geo = float(np.mean(np.abs(explanation.geo_values)))
    ax.set_title(f"SpatialSHAP components (mean |GEO main|={mean_geo:.3g})")
    ax.legend()
    return fig, ax
