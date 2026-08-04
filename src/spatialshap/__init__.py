"""Public package interface for SpatialSHAP."""

from spatialshap._explainer import Explainer, explain
from spatialshap._explanation import SpatialExplanation
from spatialshap._geo_coalitions import GeoDecompositionDiagnostics
from spatialshap._geo_explainer import GeoExplainer, explain_geo
from spatialshap._geo_explanation import GeoExplanation
from spatialshap._references import (
    GlobalReference,
    KernelReference,
    KNNReference,
    ReferenceDiagnostics,
)
from spatialshap._version import __version__

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
    "explain",
    "explain_geo",
    "__version__",
]
