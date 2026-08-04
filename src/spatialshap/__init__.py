"""Public package interface for SpatialSHAP."""

from spatialshap._explainer import Explainer, explain
from spatialshap._explanation import SpatialExplanation
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
    "GlobalReference",
    "KernelReference",
    "KNNReference",
    "ReferenceDiagnostics",
    "explain",
    "__version__",
]
