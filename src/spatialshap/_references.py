"""Spatial reference distributions for conditional Shapley values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class ReferenceDiagnostics:
    """Auditable diagnostics for one focal reference distribution."""

    n_background: int
    n_positive: int
    effective_n: float
    mean_distance: float
    max_distance: float
    weight_concentration: float
    fallback_used: bool = False

    def as_dict(self) -> dict[str, float | int | bool]:
        """Return a flat representation suitable for tabular output."""

        return {
            "reference_n_background": self.n_background,
            "reference_n_positive": self.n_positive,
            "reference_effective_n": self.effective_n,
            "reference_mean_distance": self.mean_distance,
            "reference_max_distance": self.max_distance,
            "reference_weight_concentration": self.weight_concentration,
            "reference_fallback_used": self.fallback_used,
        }


class Reference(Protocol):
    """Protocol implemented by all reference-distribution strategies."""

    def weights(
        self,
        focal_geometry: FloatArray | None,
        background_geometry: FloatArray | None,
        n_background: int,
    ) -> tuple[FloatArray, ReferenceDiagnostics]:
        """Return normalized weights and diagnostics for one focal observation."""


def _validate_geometry(
    focal_geometry: FloatArray | None,
    background_geometry: FloatArray | None,
    n_background: int,
) -> tuple[FloatArray, FloatArray]:
    if focal_geometry is None or background_geometry is None:
        raise ValueError(
            "This reference requires focal and background geometry coordinates."
        )
    focal = np.asarray(focal_geometry, dtype=float)
    background = np.asarray(background_geometry, dtype=float)
    if focal.shape != (2,):
        raise ValueError("focal_geometry must contain exactly two coordinates.")
    if background.shape != (n_background, 2):
        raise ValueError(
            "background_geometry must have shape (n_background, 2)."
        )
    if not np.isfinite(focal).all() or not np.isfinite(background).all():
        raise ValueError("Geometry coordinates must be finite.")
    return focal, background


def _diagnostics(weights: FloatArray, distances: FloatArray) -> ReferenceDiagnostics:
    positive = weights > 0
    n_positive = int(np.count_nonzero(positive))
    effective_n = float(1.0 / np.sum(np.square(weights)))
    if n_positive:
        local_distances = distances[positive]
        local_weights = weights[positive]
        conditional_weights = local_weights / local_weights.sum()
        mean_distance = float(np.dot(conditional_weights, local_distances))
        max_distance = float(local_distances.max())
    else:  # pragma: no cover - guarded by all reference constructors
        mean_distance = float("nan")
        max_distance = float("nan")
    return ReferenceDiagnostics(
        n_background=int(weights.size),
        n_positive=n_positive,
        effective_n=effective_n,
        mean_distance=mean_distance,
        max_distance=max_distance,
        weight_concentration=float(np.max(weights)),
    )


@dataclass(frozen=True)
class GlobalReference:
    """Use every background observation with equal weight."""

    def weights(
        self,
        focal_geometry: FloatArray | None,
        background_geometry: FloatArray | None,
        n_background: int,
    ) -> tuple[FloatArray, ReferenceDiagnostics]:
        if n_background <= 0:
            raise ValueError("n_background must be positive.")
        weights: FloatArray = np.full(
            n_background, 1.0 / n_background, dtype=float
        )
        distances: FloatArray = np.zeros(n_background, dtype=float)
        return weights, _diagnostics(weights, distances)


@dataclass(frozen=True)
class KernelReference:
    """Construct a normalized distance-kernel reference distribution."""

    bandwidth: float
    kernel: str = "bisquare"

    def __post_init__(self) -> None:
        if not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("bandwidth must be a finite positive number.")
        if self.kernel not in {"bisquare", "gaussian", "exponential"}:
            raise ValueError(
                "kernel must be one of 'bisquare', 'gaussian', or 'exponential'."
            )

    def weights(
        self,
        focal_geometry: FloatArray | None,
        background_geometry: FloatArray | None,
        n_background: int,
    ) -> tuple[FloatArray, ReferenceDiagnostics]:
        focal, background = _validate_geometry(
            focal_geometry,
            background_geometry,
            n_background,
        )
        distances: FloatArray = np.linalg.norm(background - focal, axis=1)
        ratio = distances / self.bandwidth
        raw: FloatArray
        if self.kernel == "bisquare":
            raw = np.where(ratio < 1.0, np.square(1.0 - np.square(ratio)), 0.0)
        elif self.kernel == "gaussian":
            raw = np.exp(-0.5 * np.square(ratio))
        else:
            raw = np.exp(-ratio)
        total = float(raw.sum())
        if total <= 0:
            raise ValueError(
                "KernelReference selected no positive-weight background rows; "
                "increase the bandwidth or use KNNReference."
            )
        weights: FloatArray = np.asarray(raw / total, dtype=float)
        return weights, _diagnostics(weights, distances)


@dataclass(frozen=True)
class KNNReference:
    """Use the nearest ``k`` background observations as the reference."""

    k: int
    distance_weighted: bool = False
    power: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("k must be a positive integer.")
        if not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("power must be a finite positive number.")

    def weights(
        self,
        focal_geometry: FloatArray | None,
        background_geometry: FloatArray | None,
        n_background: int,
    ) -> tuple[FloatArray, ReferenceDiagnostics]:
        focal, background = _validate_geometry(
            focal_geometry,
            background_geometry,
            n_background,
        )
        if self.k > n_background:
            raise ValueError("k cannot exceed the number of background rows.")
        distances: FloatArray = np.linalg.norm(background - focal, axis=1)
        order = np.argsort(distances, kind="stable")[: self.k]
        raw: FloatArray = np.zeros(n_background, dtype=float)
        if self.distance_weighted:
            selected = distances[order]
            zero = selected == 0
            if np.any(zero):
                raw[order[zero]] = 1.0 / np.count_nonzero(zero)
            else:
                inverse = 1.0 / np.power(selected, self.power)
                raw[order] = inverse / inverse.sum()
        else:
            raw[order] = 1.0 / self.k
        weights: FloatArray = np.asarray(raw / raw.sum(), dtype=float)
        return weights, _diagnostics(weights, distances)
