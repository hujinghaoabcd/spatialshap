import numpy as np
import pandas as pd
import pytest

from spatialshap import KNNReference
from spatialshap.estimand_comparison import (
    GeographicEstimandComparison,
    compare_geographic_estimands,
    exact_coordinate_group_coalition_values,
    exact_reference_switch_joint_coalition_values,
)


def joint_linear(values):
    array = np.asarray(values)
    feature = array[:, 0]
    coordinate_x = array[:, 1]
    return 1.0 + 2.0 * feature + 3.0 * coordinate_x


def test_exact_games_have_distinct_closed_form_geo_meanings():
    background = np.array([[0.0], [2.0]])
    background_geometry = np.array([[0.0, 0.0], [2.0, 0.0]])
    target = np.array([4.0])
    target_geometry = np.array([3.0, 0.0])
    global_weights = np.array([0.5, 0.5])
    local_weights = np.array([0.0, 1.0])

    coordinate = exact_coordinate_group_coalition_values(
        joint_linear,
        target,
        target_geometry,
        background,
        background_geometry,
        global_weights,
    )
    reference = exact_reference_switch_joint_coalition_values(
        joint_linear,
        target,
        target_geometry,
        background,
        global_weights,
        local_weights,
    )

    np.testing.assert_allclose(coordinate, [6.0, 12.0, 12.0, 18.0])
    np.testing.assert_allclose(reference, [12.0, 18.0, 14.0, 18.0])


def test_no_coordinate_effect_only_triggers_reference_switch_geo():
    def feature_only(values):
        array = np.asarray(values)
        return 2.0 * array[:, 0]

    background = np.array([[0.0], [2.0]])
    background_geometry = np.array([[0.0, 0.0], [2.0, 0.0]])
    comparison = compare_geographic_estimands(
        feature_only,
        pd.DataFrame([[4.0]], columns=["feature"]),
        background=pd.DataFrame(background, columns=["feature"]),
        geometry=np.array([[3.0, 0.0]]),
        background_geometry=background_geometry,
        reference=KNNReference(k=1),
    )
    assert comparison.coordinate_group.geo_values[0] == pytest.approx(0.0)
    assert comparison.coordinate_group.interaction_values[0, 0] == pytest.approx(
        0.0
    )
    assert comparison.reference_switch.geo_values[0] == pytest.approx(2.0)
    assert comparison.reference_switch.interaction_values[0, 0] == pytest.approx(
        -2.0
    )


def test_coordinate_only_effect_only_triggers_coordinate_group_geo():
    def coordinate_only(values):
        array = np.asarray(values)
        return 3.0 * array[:, 1]

    background = pd.DataFrame([[0.0], [2.0]], columns=["feature"])
    background_geometry = np.array([[0.0, 0.0], [2.0, 0.0]])
    comparison = compare_geographic_estimands(
        coordinate_only,
        pd.DataFrame([[4.0]], columns=["feature"]),
        background=background,
        geometry=np.array([[3.0, 0.0]]),
        background_geometry=background_geometry,
        reference=KNNReference(k=1),
    )
    assert comparison.coordinate_group.geo_values[0] == pytest.approx(6.0)
    assert comparison.reference_switch.geo_values[0] == pytest.approx(0.0)
    np.testing.assert_allclose(comparison.reference_switch.primary_values, 0.0)
    np.testing.assert_allclose(comparison.reference_switch.interaction_values, 0.0)


def test_high_level_comparison_matches_linear_closed_form():
    background = pd.DataFrame([[0.0], [2.0]], columns=["feature"])
    comparison = compare_geographic_estimands(
        joint_linear,
        pd.DataFrame([[4.0]], columns=["feature"]),
        background=background,
        geometry=np.array([[3.0, 0.0]]),
        background_geometry=np.array([[0.0, 0.0], [2.0, 0.0]]),
        reference=KNNReference(k=1),
    )

    assert isinstance(comparison, GeographicEstimandComparison)
    np.testing.assert_allclose(comparison.coordinate_group.primary_values, [[6.0]])
    np.testing.assert_allclose(comparison.coordinate_group.geo_values, [6.0])
    np.testing.assert_allclose(comparison.coordinate_group.interaction_values, 0.0)
    np.testing.assert_allclose(comparison.coordinate_group.base_values, [6.0])
    np.testing.assert_allclose(comparison.reference_switch.primary_values, [[6.0]])
    np.testing.assert_allclose(comparison.reference_switch.geo_values, [2.0])
    np.testing.assert_allclose(
        comparison.reference_switch.interaction_values,
        [[-2.0]],
    )
    np.testing.assert_allclose(comparison.reference_switch.base_values, [12.0])
    np.testing.assert_allclose(comparison.coordinate_group.predictions, [18.0])
    np.testing.assert_allclose(comparison.reference_switch.predictions, [18.0])
    np.testing.assert_allclose(comparison.coordinate_group.additivity_error, 0.0)
    np.testing.assert_allclose(comparison.reference_switch.additivity_error, 0.0)
    assert comparison.coordinate_group.decomposition_diagnostics[
        0
    ].weighted_residual_rmse < 1e-12
    assert comparison.reference_switch.decomposition_diagnostics[
        0
    ].weighted_residual_rmse < 1e-12


def test_comparison_tables_summary_and_read_only_arrays():
    comparison = compare_geographic_estimands(
        joint_linear,
        pd.DataFrame([[4.0]], columns=["feature"]),
        background=pd.DataFrame([[0.0], [2.0]], columns=["feature"]),
        geometry=np.array([[3.0, 0.0]]),
        background_geometry=np.array([[0.0, 0.0], [2.0, 0.0]]),
        reference=KNNReference(k=1),
    )
    table = comparison.component_table()
    assert {
        "baseline",
        "primary",
        "geo_main",
        "geo_interaction",
        "joint_feature_shapley",
        "joint_geo_shapley",
    } == set(table.index)
    frame = comparison.to_frame()
    assert frame.loc[0, "difference__geo_main"] == pytest.approx(-4.0)
    assert frame.loc[0, "difference__base"] == pytest.approx(6.0)
    assert "difference__geo_interaction__feature" in frame.columns
    assert "GeographicEstimandComparison" in comparison.summary()
    with pytest.raises(ValueError):
        comparison.data[0, 0] = 0.0
    with pytest.raises(ValueError):
        comparison.coordinate_group.primary_values[0, 0] = 0.0


def test_estimand_comparison_validates_contracts():
    background = pd.DataFrame([[0.0], [2.0]], columns=["feature"])
    with pytest.raises(ValueError, match="same feature count"):
        compare_geographic_estimands(
            joint_linear,
            np.ones((1, 2)),
            background=background,
            geometry=np.array([[3.0, 0.0]]),
            background_geometry=np.array([[0.0, 0.0], [2.0, 0.0]]),
            reference=KNNReference(k=1),
        )
    with pytest.raises(ValueError, match="required"):
        compare_geographic_estimands(
            joint_linear,
            background.iloc[:1],
            background=background,
            geometry=None,
            background_geometry=np.array([[0.0, 0.0], [2.0, 0.0]]),
            reference=KNNReference(k=1),
        )
    with pytest.raises(ValueError, match="positive integer"):
        compare_geographic_estimands(
            joint_linear,
            background.iloc[:1],
            background=background,
            geometry=np.array([[3.0, 0.0]]),
            background_geometry=np.array([[0.0, 0.0], [2.0, 0.0]]),
            reference=KNNReference(k=1),
            max_exact_features=0,
        )
    with pytest.raises(ValueError, match="coordinate count"):
        exact_coordinate_group_coalition_values(
            joint_linear,
            np.array([1.0]),
            np.array([1.0]),
            np.array([[0.0], [2.0]]),
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            np.array([0.5, 0.5]),
        )
