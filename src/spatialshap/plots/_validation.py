"""Validation and sensitivity plots."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from spatialshap._geo_explanation import GeoExplanation
from spatialshap.plots._summary import _plt
from spatialshap.validation import FloatArray, LinearReferenceSwitchTruth


def bandwidth_profile(
    profile: pd.DataFrame,
    *,
    metric: str = "mean_abs_geo_main",
    log_x: bool = False,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot one metric from :func:`spatialshap.bandwidth_sensitivity`."""

    if "bandwidth" not in profile.columns:
        raise ValueError("profile must contain a 'bandwidth' column.")
    if metric not in profile.columns:
        raise KeyError(metric)
    bandwidth = np.asarray(profile["bandwidth"], dtype=float)
    values = np.asarray(profile[metric], dtype=float)
    if bandwidth.ndim != 1 or values.shape != bandwidth.shape or bandwidth.size == 0:
        raise ValueError("profile columns must be non-empty one-dimensional arrays.")
    if not np.isfinite(bandwidth).all() or np.any(bandwidth <= 0):
        raise ValueError("bandwidth values must be finite and positive.")
    if not np.isfinite(values).all():
        raise ValueError("the selected metric must contain finite values.")

    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    order = np.argsort(bandwidth, kind="stable")
    ax.plot(bandwidth[order], values[order], marker="o")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("SpatialSHAP bandwidth sensitivity")
    ax.grid(axis="y", alpha=0.25)
    return fig, ax


def _feature_index(explanation: GeoExplanation, feature: str | int | None) -> int:
    if feature is None:
        raise ValueError("feature is required for this recovery component.")
    if isinstance(feature, str):
        try:
            return explanation.feature_names.index(feature)
        except ValueError as exc:
            raise KeyError(feature) from exc
    index = int(feature)
    if index < 0 or index >= explanation.shape[1]:
        raise IndexError(index)
    return index


def recovery_scatter(
    explanation: GeoExplanation,
    truth: LinearReferenceSwitchTruth,
    *,
    component: str = "geo_main",
    feature: str | int | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot estimated values against analytic truth for one component."""

    if explanation.shape != truth.shape:
        raise ValueError("explanation and truth must have the same shape.")
    if component == "geo_main":
        estimated: FloatArray = explanation.geo_values
        target: FloatArray = truth.geo_values
        label = "GEO main"
    elif component == "joint_geo_shapley":
        estimated = explanation.shapley_values[:, -1]
        target = truth.shapley_values[:, -1]
        label = "Joint GEO Shapley"
    elif component in {"primary", "geo_interaction", "joint_feature_shapley"}:
        index = _feature_index(explanation, feature)
        name = explanation.feature_names[index]
        if component == "primary":
            estimated = explanation.primary_values[:, index]
            target = truth.primary_values[:, index]
            label = f"Primary: {name}"
        elif component == "geo_interaction":
            estimated = explanation.interaction_values[:, index]
            target = truth.interaction_values[:, index]
            label = f"GEO interaction: {name}"
        else:
            estimated = explanation.shapley_values[:, index]
            target = truth.shapley_values[:, index]
            label = f"Joint Shapley: {name}"
    else:
        raise ValueError(
            "component must be 'primary', 'geo_main', 'geo_interaction', "
            "'joint_feature_shapley', or 'joint_geo_shapley'."
        )

    lower = float(min(np.min(target), np.min(estimated)))
    upper = float(max(np.max(target), np.max(estimated)))
    if np.isclose(lower, upper):
        padding = max(1.0, abs(lower) * 0.05)
        lower -= padding
        upper += padding

    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
    else:
        fig = ax.figure
    ax.scatter(target, estimated, s=34, alpha=0.8)
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Analytic truth")
    ax.set_ylabel("Estimated contribution")
    ax.set_title(f"SpatialSHAP recovery — {label}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    return fig, ax
