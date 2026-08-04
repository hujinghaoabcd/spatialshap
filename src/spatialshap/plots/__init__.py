"""Public plotting interface for SpatialSHAP."""

from spatialshap.plots._local import waterfall
from spatialshap.plots._maps import (
    baseline_map,
    effect_map,
    effective_reference_map,
)
from spatialshap.plots._summary import bar, beeswarm

__all__ = [
    "bar",
    "beeswarm",
    "waterfall",
    "effect_map",
    "baseline_map",
    "effective_reference_map",
]
