"""Public plotting interface for SpatialSHAP."""

from spatialshap.plots._estimand import (
    estimand_difference_map,
    estimand_scatter,
)
from spatialshap.plots._geo import (
    component_bar,
    geo_effect_map,
    interaction_map,
    structure_diagnostic_map,
)
from spatialshap.plots._local import waterfall
from spatialshap.plots._maps import (
    baseline_map,
    effect_map,
    effective_reference_map,
)
from spatialshap.plots._summary import bar, beeswarm
from spatialshap.plots._validation import bandwidth_profile, recovery_scatter

__all__ = [
    "bar",
    "beeswarm",
    "waterfall",
    "effect_map",
    "baseline_map",
    "effective_reference_map",
    "component_bar",
    "geo_effect_map",
    "interaction_map",
    "structure_diagnostic_map",
    "bandwidth_profile",
    "recovery_scatter",
    "estimand_scatter",
    "estimand_difference_map",
]
