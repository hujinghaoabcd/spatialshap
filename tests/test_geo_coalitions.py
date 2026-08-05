import numpy as np
import pytest

from spatialshap._coalitions import exact_shapley_values
from spatialshap._geo_coalitions import (
    exact_geo_decomposition,
    exact_joint_coalition_values,
    feature_pair_second_differences,
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
    assert diagnostics.weighted_residual_rmse < 1e-12
    assert diagnostics.relative_residual_norm < 1e-12
    assert diagnostics.max_abs_feature_pair_second_difference == 0.0
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
        assert diagnostics.weighted_residual_rmse >= 0.0
        assert diagnostics.relative_residual_norm >= 0.0
        assert diagnostics.max_abs_coalition_residual >= 0.0


def _represented_geo_game(n_features):
    n_players = n_features + 1
    geo_bit = 1 << n_features
    primary = np.linspace(0.5, 1.5, n_features)
    geo = -0.75
    interactions = np.linspace(-0.4, 0.6, n_features)
    values = np.full(1 << n_players, 2.0)
    for mask in range(1 << n_players):
        geo_present = bool(mask & geo_bit)
        for feature in range(n_features):
            present = bool(mask & (1 << feature))
            if present:
                values[mask] += primary[feature]
            if present and geo_present:
                values[mask] += interactions[feature]
        if geo_present:
            values[mask] += geo
    return values


def test_represented_game_has_zero_structure_residuals():
    values = _represented_geo_game(3)
    _, _, _, diagnostics = exact_geo_decomposition(values, 3)
    assert diagnostics.weighted_residual_rmse < 1e-12
    assert diagnostics.relative_residual_norm < 1e-12
    assert diagnostics.max_abs_coalition_residual < 1e-12
    assert diagnostics.mean_abs_feature_pair_second_difference < 1e-12
    assert diagnostics.max_abs_feature_pair_second_difference < 1e-12
    np.testing.assert_allclose(
        feature_pair_second_differences(values, 3),
        0.0,
        atol=1e-12,
    )


def test_feature_pair_term_is_exposed_as_unrepresented_structure():
    n_features = 2
    values = _represented_geo_game(n_features)
    interaction_strength = 3.25
    for mask in range(values.size):
        if mask & 1 and mask & 2:
            values[mask] += interaction_strength

    _, _, _, diagnostics = exact_geo_decomposition(values, n_features)
    differences = feature_pair_second_differences(values, n_features)
    np.testing.assert_allclose(differences, interaction_strength)
    assert diagnostics.weighted_residual_rmse > 0.0
    assert diagnostics.relative_residual_norm > 0.0
    assert diagnostics.max_abs_coalition_residual > 0.0
    assert diagnostics.mean_abs_feature_pair_second_difference == pytest.approx(
        interaction_strength
    )
    assert diagnostics.max_abs_feature_pair_second_difference == pytest.approx(
        interaction_strength
    )
    assert diagnostics.constraint_error < 1e-12
    assert diagnostics.shapley_equivalence_error < 1e-12


def test_geo_feature_terms_do_not_trigger_feature_pair_diagnostic():
    values = _represented_geo_game(2)
    differences = feature_pair_second_differences(values, 2)
    np.testing.assert_allclose(differences, 0.0, atol=1e-12)
    _, _, _, diagnostics = exact_geo_decomposition(values, 2)
    assert diagnostics.weighted_residual_rmse < 1e-12
    assert diagnostics.max_abs_feature_pair_second_difference < 1e-12


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


def test_feature_pair_second_differences_validate_contract():
    with pytest.raises(ValueError, match="positive"):
        feature_pair_second_differences(np.ones(2), 0)
    with pytest.raises(ValueError, match="shape"):
        feature_pair_second_differences(np.ones(4), 2)
    with pytest.raises(ValueError, match="finite"):
        feature_pair_second_differences(
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan]),
            2,
        )
