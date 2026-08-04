"""Global attribution summary plots."""

from __future__ import annotations

from typing import Any

import numpy as np

from spatialshap._explanation import SpatialExplanation


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Matplotlib is required; install spatialshap[plot]."
        ) from exc
    return plt


def bar(
    explanation: SpatialExplanation,
    *,
    max_display: int = 12,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot ranked global mean absolute contributions."""

    if max_display <= 0:
        raise ValueError("max_display must be positive.")
    plt = _plt()
    series = explanation.mean_abs_values.iloc[:max_display].sort_values()
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.38 * len(series) + 1.5)))
    else:
        fig = ax.figure
    ax.barh(series.index, series.to_numpy())
    ax.set_xlabel("Mean absolute contribution")
    ax.set_ylabel("Feature")
    ax.set_title("SpatialSHAP global importance")
    return fig, ax


def beeswarm(
    explanation: SpatialExplanation,
    *,
    max_display: int = 12,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot a deterministic beeswarm-style attribution summary."""

    if max_display <= 0:
        raise ValueError("max_display must be positive.")
    plt = _plt()
    order = np.argsort(np.mean(np.abs(explanation.values), axis=0))[::-1]
    order = order[:max_display]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(order) + 1.5)))
    else:
        fig = ax.figure
    rng = np.random.default_rng(0)
    for y_position, feature_index in enumerate(order):
        jitter = rng.uniform(-0.16, 0.16, size=explanation.shape[0])
        ax.scatter(
            explanation.values[:, feature_index],
            y_position + jitter,
            c=explanation.data[:, feature_index],
            cmap="coolwarm",
            s=22,
            alpha=0.8,
            edgecolors="none",
        )
    ax.axvline(0.0, linewidth=0.8, color="black")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([explanation.feature_names[int(i)] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Contribution")
    ax.set_title("SpatialSHAP contribution distribution")
    return fig, ax
