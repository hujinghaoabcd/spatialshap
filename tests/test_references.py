import numpy as np
import pytest

from spatialshap import GlobalReference, KernelReference, KNNReference


def test_global_reference_is_equal_weighted():
    weights, diagnostics = GlobalReference().weights(None, None, 4)
    np.testing.assert_allclose(weights, np.full(4, 0.25))
    assert diagnostics.effective_n == pytest.approx(4.0)
    assert diagnostics.n_positive == 4


def test_bisquare_kernel_has_compact_support():
    background = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    weights, diagnostics = KernelReference(
        bandwidth=2.0,
        kernel="bisquare",
    ).weights(np.array([0.0, 0.0]), background, 3)
    assert weights[2] == 0.0
    assert weights.sum() == pytest.approx(1.0)
    assert diagnostics.n_positive == 2


def test_knn_reference_selects_stable_nearest_rows():
    background = np.array([[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    weights, diagnostics = KNNReference(k=2).weights(
        np.array([0.0, 0.0]), background, 3
    )
    np.testing.assert_allclose(weights, [0.5, 0.5, 0.0])
    assert diagnostics.effective_n == pytest.approx(2.0)


def test_kernel_reference_rejects_empty_support():
    background = np.array([[10.0, 0.0], [20.0, 0.0]])
    with pytest.raises(ValueError, match="selected no positive-weight"):
        KernelReference(bandwidth=1.0).weights(
            np.array([0.0, 0.0]), background, 2
        )


def test_reference_validation_and_continuous_kernels():
    with pytest.raises(ValueError, match="bandwidth"):
        KernelReference(bandwidth=0.0)
    with pytest.raises(ValueError, match="kernel"):
        KernelReference(bandwidth=1.0, kernel="unknown")
    with pytest.raises(ValueError, match="requires focal"):
        KernelReference(bandwidth=1.0).weights(None, None, 2)

    background = np.array([[0.0, 0.0], [2.0, 0.0]])
    for kernel in ("gaussian", "exponential"):
        weights, _ = KernelReference(bandwidth=1.0, kernel=kernel).weights(
            np.array([0.0, 0.0]), background, 2
        )
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights > 0)


def test_distance_weighted_knn_handles_exact_matches():
    background = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    weights, diagnostics = KNNReference(k=3, distance_weighted=True).weights(
        np.array([0.0, 0.0]), background, 3
    )
    np.testing.assert_allclose(weights, [0.5, 0.5, 0.0])
    assert diagnostics.n_positive == 2


def test_knn_validation_is_explicit():
    with pytest.raises(ValueError, match="positive integer"):
        KNNReference(k=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        KNNReference(k=3).weights(
            np.array([0.0, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0]]), 2
        )
