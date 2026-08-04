import numpy as np
import pandas as pd
import pytest

from spatialshap import (
    GeoExplainer,
    KernelReference,
    bandwidth_sensitivity,
    geo_recovery_table,
    linear_reference_switch_truth,
    recovery_metrics,
)


def test_linear_geo_decomposition_matches_analytic_truth():
    background = pd.DataFrame(
        {
            "density": [0.0, 1.0, 2.0, 4.0, 6.0],
            "access": [2.0, 1.5, 1.0, 0.5, -0.5],
        }
    )
    background_geometry = np.column_stack(
        [np.linspace(0.0, 8.0, len(background)), np.zeros(len(background))]
    )
    X = background.iloc[[0, 2, 4]].copy()
    geometry = background_geometry[[0, 2, 4]]
    coefficients = np.array([1.7, -0.8])
    intercept = 2.5

    def predict(values):
        return intercept + np.asarray(values) @ coefficients

    reference = KernelReference(bandwidth=2.5, kernel="gaussian")
    local_weights = np.vstack(
        [
            reference.weights(point, background_geometry, len(background))[0]
            for point in geometry
        ]
    )
    truth = linear_reference_switch_truth(
        X,
        coefficients,
        background=background,
        local_weights=local_weights,
        intercept=intercept,
    )
    result = GeoExplainer(
        predict,
        background,
        reference=reference,
        background_geometry=background_geometry,
    )(X, geometry=geometry)

    np.testing.assert_allclose(result.primary_values, truth.primary_values, atol=1e-10)
    np.testing.assert_allclose(result.geo_values, truth.geo_values, atol=1e-10)
    np.testing.assert_allclose(
        result.interaction_values,
        truth.interaction_values,
        atol=1e-10,
    )
    np.testing.assert_allclose(result.shapley_values, truth.shapley_values, atol=1e-10)
    np.testing.assert_allclose(truth.additivity_error, 0.0, atol=1e-12)

    table = geo_recovery_table(result, truth)
    assert set(table.index) == {
        "primary",
        "geo_main",
        "geo_interaction",
        "joint_feature_shapley",
        "joint_geo_shapley",
    }
    assert float(table["rmse"].max()) < 1e-10
    assert float(table["sign_agreement"].min()) == pytest.approx(1.0)


def test_zero_coefficient_feature_has_zero_truth_components():
    background = np.array(
        [
            [0.0, -10.0],
            [1.0, 0.0],
            [2.0, 20.0],
        ]
    )
    X = np.array([[1.5, 100.0]])
    local_weights = np.array([[0.0, 0.0, 1.0]])
    truth = linear_reference_switch_truth(
        X,
        np.array([2.0, 0.0]),
        background=background,
        local_weights=local_weights,
    )
    assert truth.primary_values[0, 1] == pytest.approx(0.0)
    assert truth.interaction_values[0, 1] == pytest.approx(0.0)


def test_recovery_metrics_handle_exact_and_invalid_inputs():
    exact = recovery_metrics(np.array([1.0, -2.0, 0.0]), np.array([1.0, -2.0, 0.0]))
    assert exact.rmse == pytest.approx(0.0)
    assert exact.correlation == pytest.approx(1.0)
    assert exact.sign_agreement == pytest.approx(1.0)

    constant = recovery_metrics(np.ones(3), np.ones(3))
    assert constant.correlation == pytest.approx(1.0)

    with pytest.raises(ValueError, match="same shape"):
        recovery_metrics(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="non-negative"):
        recovery_metrics(np.ones(2), np.ones(2), zero_tolerance=-1.0)


def test_bandwidth_sensitivity_approaches_global_reference():
    coordinates = np.column_stack([np.linspace(0.0, 10.0, 21), np.zeros(21)])
    background = pd.DataFrame(
        {
            "gradient": coordinates[:, 0],
            "constant": np.ones(21),
        }
    )
    X = background.iloc[[0, -1]].copy()
    geometry = coordinates[[0, -1]]

    def predict(values):
        array = np.asarray(values)
        return 1.2 * array[:, 0] + 0.4 * array[:, 1]

    profile = bandwidth_sensitivity(
        predict,
        X,
        background=background,
        geometry=geometry,
        background_geometry=coordinates,
        bandwidths=[0.7, 2.0, 1000.0],
    )
    assert list(profile["bandwidth"]) == [0.7, 2.0, 1000.0]
    assert profile.loc[2, "mean_abs_geo_main"] < profile.loc[0, "mean_abs_geo_main"]
    assert (
        profile.loc[2, "mean_abs_geo_interaction"]
        < profile.loc[0, "mean_abs_geo_interaction"]
    )
    assert float(profile["max_additivity_error"].max()) < 1e-10
    assert float(profile["max_shapley_equivalence_error"].max()) < 1e-10
    assert profile.loc[2, "mean_effective_reference_n"] > profile.loc[0, "mean_effective_reference_n"]


def test_linear_truth_rejects_invalid_weight_contracts():
    with pytest.raises(ValueError, match="local_weights"):
        linear_reference_switch_truth(
            np.ones((1, 2)),
            np.ones(2),
            background=np.ones((3, 2)),
            local_weights=np.ones((1, 2)),
        )
    with pytest.raises(ValueError, match="positive total"):
        linear_reference_switch_truth(
            np.ones((1, 2)),
            np.ones(2),
            background=np.ones((3, 2)),
            local_weights=np.zeros((1, 3)),
        )
