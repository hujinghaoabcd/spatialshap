import numpy as np
import pytest

from spatialshap._coalitions import exact_shapley_values
from spatialshap._geo_coalitions import (
    exact_geo_decomposition,
    exact_joint_coalition_values,
    shapley_kernel_weight,
)


def test_joint_coalitions_reuse_predictions_and_switch_weights():
    calls = []

    def predict(values):
        calls.append(values.copy())
        return values[:, 0] + 2.0 * values[:, 1]

    background = np.array([[0.0, 0.0], [1.0, 3.0], [2.0, 5.0]])
    target = np.array([4.0, 7.0])
    global_weights = np.array([1 / 3, 1 / 3, 1 / 3])
    local_weights = np.array([0.0, 1.0, 0.0])
    values = exact_joint_coalition_values(
        predict,
        target,
        background,
        global_weights,
        local_weights,
    )
    assert len(calls) == 4
    assert values.shape == (8,)
    assert values[0] == pytest.approx(np.mean([0.0, 7.0, 12.0]))
    assert values[4] == pytest.approx(7.0)
    assert values[3] == pytest.approx(18.0)
    assert values[7] == pytest.approx(18.0)


def test_one_feature_decomposition_has_closed_form():
    values = np.array([10.0, 15.0, 12.0, 15.0])
    primary, geo, interaction, diagnostics = exact_geo_decomposition(values, 1)
    np.testing.assert_allclose(primary, [5.0])
    assert geo == pytest.approx(2.0)
    np.testing.assert_allclose(interaction, [-2.0])
    assert diagnostics.constraint_error < 1e-12
    assert diagnostics.shapley_equivalence_error < 1e-12


def test_decomposition_redistributes_to_exact_joint_shapley():
    rng = np.random.default_rng(31)
    for n_features in (1, 2, 3, 4):
        n_players = n_features + 1
        values = rng.normal(size=1 << n_players)
        primary, geo, interactions, diagnostics = exact_geo_decomposition(
            values,
            n_features,
        )
        redistributed = np.concatenate(
            [
                primary + interactions / 2,
                [geo + interactions.sum() / 2],
            ]
        )
        expected = exact_shapley_values(values, n_players)
        np.testing.assert_allclose(redistributed, expected, atol=1e-11)
        assert diagnostics.design_rank == 2 * n_features + 1


def test_shapley_kernel_rejects_endpoints():
    assert shapley_kernel_weight(3, 1) > 0
    with pytest.raises(ValueError):
        shapley_kernel_weight(3, 0)
    with pytest.raises(ValueError):
        shapley_kernel_weight(3, 3)


def test_joint_coalitions_validate_weights_and_predictions():
    background = np.ones((3, 2))
    target = np.ones(2)
    with pytest.raises(ValueError):
        exact_joint_coalition_values(
            lambda values: np.ones(len(values)),
            target,
            background,
            np.ones(2),
            np.ones(3),
        )
    with pytest.raises(ValueError):
        exact_joint_coalition_values(
            lambda values: np.ones((len(values), 1)),
            target,
            background,
            np.ones(3),
            np.ones(3),
        )
