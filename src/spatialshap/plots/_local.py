"""Local attribution plots."""

from __future__ import annotations

from typing import Any

import numpy as np

from spatialshap._explanation import SpatialExplanation
from spatialshap.plots._summary import _plt


def waterfall(
    explanation: SpatialExplanation,
    *,
    max_display: int = 12,
    ax: Any | None = None,
):
    """Plot one observation's additive reconstruction."""

    if explanation.shape[0] != 1:
        raise ValueError("waterfall requires an explanation containing one row.")
    if max_display <= 0:
        raise ValueError("max_display must be positive.")
    plt = _plt()
    contributions = explanation.values[0]
    order = np.argsort(np.abs(contributions))[::-1][:max_display]
    names = [explanation.feature_names[int(i)] for i in order]
    selected = contributions[order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.42 * len(order) + 1.8)))
    else:
        fig = ax.figure
    y = np.arange(len(order))
    ax.barh(y, selected)
    ax.axvline(0.0, linewidth=0.8, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution")
    ax.set_title(
        "Local explanation: "
        f"base={explanation.base_values[0]:.4g}, "
        f"prediction={explanation.predictions[0]:.4g}"
    )
    return fig, ax
