"""Public package interface for SpatialSHAP."""

from spatialshap._explainer import Explainer, explain
from spatialshap._explanation import SpatialExplanation
from spatialshap._geo_coalitions import (
    GeoDecompositionDiagnostics,
    feature_pair_second_differences,
)
from spatialshap._geo_explainer import GeoExplainer, explain_geo
from spatialshap._geo_explanation import GeoExplanation
from spatialshap._references import (
    GlobalReference,
    KernelReference,
    KNNReference,
    ReferenceDiagnostics,
)
from spatialshap._version import __version__
from spatialshap.additive_validation import (
    AdditiveReferenceSwitchTruth,
    additive_reference_switch_truth,
)
from spatialshap.validation import (
    LinearReferenceSwitchTruth,
    RecoveryMetrics,
    bandwidth_sensitivity,
    geo_recovery_table,
    linear_reference_switch_truth,
    recovery_metrics,
)

__author__ = "Jinghao Hu"
__license__ = "MIT"

__all__ = [
    "Explainer",
    "SpatialExplanation",
    "GeoExplainer",
    "GeoExplanation",
    "GeoDecompositionDiagnostics",
    "GlobalReference",
    "KernelReference",
    "KNNReference",
    "ReferenceDiagnostics",
    "RecoveryMetrics",
    "LinearReferenceSwitchTruth",
    "AdditiveReferenceSwitchTruth",
    "feature_pair_second_differences",
    "recovery_metrics",
    "linear_reference_switch_truth",
    "additive_reference_switch_truth",
    "geo_recovery_table",
    "bandwidth_sensitivity",
    "explain",
    "explain_geo",
    "__version__",
]
