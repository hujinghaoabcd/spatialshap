import numpy as np
import pandas as pd
import pytest

from spatialshap import (
    GeoExplainer,
    KernelReference,
    additive_reference_switch_truth,
    geo_recovery_table,
)


def test_correlated_nonlinear_additive_model_matches_analytic_truth():
    axis = np.linspace(-2.5, 2.5, 31)
    background = pd.DataFrame(
        {
            "smooth": axis,
            "threshold": 0.96 * axis + 0.04 * np.sin(4.0 * axis),
        }
    )
    background_geometry = np.column_stack(
        [np.linspace(0.0, 15.0, len(background)), np.zeros(len(background))]
    )
    rows = [0, 6, 12, 18, 24, 30]
    X = background.iloc[rows].copy()
    geometry = background_geometry[rows]
    intercept = 1.3

    def smooth(values):
        return 0.7 * np.square(values) - 0.2 * values

    def threshold(values):
        return np.where(values >= 0.25, 1.8, -0.6)

    functions = (smooth, threshold)

    def predict(values):
        array = np.asarray(values)
        return intercept + smooth(array[:, 0]) + threshold(array[:, 1])

    reference = KernelReference(bandwidth=2.7, kernel="gaussian")
    local_weights = np.vstack(
        [
            reference.weights(point, background_geometry, len(background))[0]
            for point in geometry
        ]
    )
    truth = additive_reference_switch_truth(
        X,
        functions,
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
    assert float(geo_recovery_table(result, truth)["rmse"].max()) < 1e-10
    assert np.corrcoef(background["smooth"], background["threshold"])[0, 1] > 0.99


def test_additive_truth_accepts_custom_global_weights():
    background = np.array([[0.0], [1.0], [2.0]])
    X = np.array([[1.5]])
    truth = additive_reference_switch_truth(
        X,
        (lambda values: np.square(values),),
        background=background,
        local_weights=np.array([[0.0, 0.0, 1.0]]),
        global_weights=np.array([1.0, 0.0, 0.0]),
    )
    assert truth.base_values[0] == pytest.approx(0.0)
    assert truth.primary_values[0, 0] == pytest.approx(2.25)
    assert truth.geo_values[0] == pytest.approx(4.0)
    assert truth.interaction_values[0, 0] == pytest.approx(-4.0)
    assert truth.predictions[0] == pytest.approx(2.25)


def test_additive_truth_validates_feature_functions():
    X = np.ones((2, 2))
    background = np.ones((3, 2))
    weights = np.full((2, 3), 1 / 3)
    with pytest.raises(ValueError, match="feature_functions"):
        additive_reference_switch_truth(
            X,
            (lambda values: values,),
            background=background,
            local_weights=weights,
        )
    with pytest.raises(TypeError, match="callable"):
        additive_reference_switch_truth(
            X,
            (lambda values: values, 1.0),
            background=background,
            local_weights=weights,
        )
    with pytest.raises(ValueError, match="return shape"):
        additive_reference_switch_truth(
            X,
            (lambda values: values, lambda values: np.array([1.0])),
            background=background,
            local_weights=weights,
        )
    with pytest.raises(ValueError, match="non-finite"):
        additive_reference_switch_truth(
            X,
            (lambda values: values, lambda values: np.full(values.shape, np.nan)),
            background=background,
            local_weights=weights,
        )
