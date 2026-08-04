import numpy as np

from spatialshap import Explainer


def test_dummy_feature_receives_zero_value():
    background = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    target = np.array([[2.0, 100.0]])
    explanation = Explainer(
        lambda values: np.asarray(values)[:, 0] ** 2,
        background,
    )(target)
    assert abs(explanation.values[0, 1]) < 1e-12


def test_symmetric_features_receive_equal_values():
    background = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    target = np.array([[3.0, 3.0]])
    explanation = Explainer(
        lambda values: np.asarray(values)[:, 0] + np.asarray(values)[:, 1],
        background,
    )(target)
    np.testing.assert_allclose(explanation.values[0, 0], explanation.values[0, 1])


def test_additivity_for_nonlinear_interaction_model():
    background = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, -1.0]])
    target = np.array([[1.5, 3.0], [0.5, -2.0]])
    explanation = Explainer(
        lambda values: np.asarray(values)[:, 0] * np.asarray(values)[:, 1]
        + np.asarray(values)[:, 0] ** 2,
        background,
    )(target)
    np.testing.assert_allclose(explanation.additivity_error, 0.0, atol=1e-12)
