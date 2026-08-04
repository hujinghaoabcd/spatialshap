import numpy as np
import pandas as pd
import pytest

from spatialshap import Explainer, GlobalReference, KernelReference


def linear_predict(values):
    values = np.asarray(values)
    return 1.5 + 2.0 * values[:, 0] - 3.0 * values[:, 1]


def test_global_linear_explanation_matches_analytic_result():
    background = pd.DataFrame(
        [[0.0, 0.0], [2.0, 2.0], [4.0, 1.0]],
        columns=["x1", "x2"],
    )
    target = pd.DataFrame([[3.0, 4.0]], columns=background.columns)
    explanation = Explainer(
        linear_predict,
        background,
        reference=GlobalReference(),
    )(target)
    mean = background.mean(axis=0).to_numpy()
    expected = np.array([2.0 * (3.0 - mean[0]), -3.0 * (4.0 - mean[1])])
    np.testing.assert_allclose(explanation.values[0], expected)
    assert abs(explanation.additivity_error[0]) < 1e-12


def test_kernel_reference_changes_local_baseline_but_preserves_additivity():
    background = np.array([[0.0], [10.0], [20.0]])
    background_geometry = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    target = np.array([[1.0], [19.0]])
    geometry = np.array([[1.0, 0.0], [19.0, 0.0]])

    explanation = Explainer(
        lambda values: np.asarray(values)[:, 0],
        background,
        background_geometry=background_geometry,
        reference=KernelReference(bandwidth=12.0, kernel="bisquare"),
    )(target, geometry=geometry)

    assert explanation.base_values[0] < explanation.base_values[1]
    np.testing.assert_allclose(explanation.additivity_error, 0.0, atol=1e-12)


def test_dataframe_columns_must_match():
    background = pd.DataFrame([[0.0, 1.0]], columns=["a", "b"])
    target = pd.DataFrame([[0.0, 1.0]], columns=["b", "a"])
    explainer = Explainer(linear_predict, background)
    with pytest.raises(ValueError, match="columns must match"):
        explainer(target)


def test_feature_safety_limit_is_explicit():
    background = np.zeros((2, 3))
    with pytest.raises(ValueError, match="Exact enumeration"):
        Explainer(lambda x: np.zeros(len(x)), background, max_exact_features=2)


def test_model_object_and_one_shot_function_are_supported():
    from spatialshap import explain

    class SumModel:
        def predict(self, values):
            return np.asarray(values).sum(axis=1)

    background = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = explain(SumModel(), np.array([[1.0, 2.0]]), background=background)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result.additivity_error, 0.0, atol=1e-12)


def test_input_and_prediction_contracts_fail_loudly():
    with pytest.raises(TypeError, match="callable"):
        Explainer(object(), np.ones((2, 1)))
    with pytest.raises(ValueError, match="two-dimensional"):
        Explainer(lambda x: np.zeros(len(x)), np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        Explainer(lambda x: np.zeros(len(x)), np.array([[np.nan]]))
    with pytest.raises(ValueError, match="one-dimensional"):
        Explainer(lambda x: np.zeros((len(x), 1)), np.ones((2, 1)))

    explainer = Explainer(lambda x: np.asarray(x)[:, 0], np.ones((2, 1)))
    with pytest.raises(ValueError, match="same feature count"):
        explainer(np.ones((1, 2)))
    with pytest.raises(ValueError, match="geometry must have shape"):
        explainer(np.ones((1, 1)), geometry=np.ones((1, 3)))
