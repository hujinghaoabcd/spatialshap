"""Validation and sensitivity plots."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from spatialshap.plots._summary import _plt


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
