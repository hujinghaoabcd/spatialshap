import numpy as np
import pandas as pd
import pytest

from spatialshap import GeoExplainer, GlobalReference, KernelReference, explain_geo


def linear_predict(values):
    values = np.asarray(values)
    return 1.5 + 2.0 * values[:, 0] - 0.5 * values[:, 1]


def test_global_reference_is_exact_null_geo_player():
    background = pd.DataFrame(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]],
        columns=["a", "b"],
    )
    X = background.iloc[:2]
    result = GeoExplainer(
        linear_predict,
        background,
        reference=GlobalReference(),
    )(X)
    np.testing.assert_allclose(result.geo_values, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.interaction_values, 0.0, atol=1e-12)
    expected = (X.to_numpy() - background.mean().to_numpy()) * np.array(
        [2.0, -0.5]
    )
    np.testing.assert_allclose(result.primary_values, expected, atol=1e-12)
    np.testing.assert_allclose(result.additivity_error, 0.0, atol=1e-12)


def test_kernel_reference_produces_auditable_four_component_result():
    background = pd.DataFrame(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [4.0, 8.0]],
        columns=["a", "b"],
    )
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [8.0, 0.0]])
    result = GeoExplainer(
        linear_predict,
        background,
        reference=KernelReference(3.0, kernel="gaussian"),
        background_geometry=coords,
    )(background.iloc[:3], geometry=coords[:3])
    assert result.shape == (3, 2)
    assert result.feature_names == ("a", "b")
    assert result.metadata["geo_player"] == "reference_switch"
    np.testing.assert_allclose(result.additivity_error, 0.0, atol=1e-10)
    np.testing.assert_allclose(
        result.shapley_values.sum(axis=1) + result.base_values,
        result.predictions,
        atol=1e-10,
    )
    assert all(item.effective_n > 0 for item in result.reference_diagnostics)
    assert all(
        item.shapley_equivalence_error < 1e-10
        for item in result.decomposition_diagnostics
    )


def test_one_shot_api_and_column_validation():
    background = pd.DataFrame(
        [[0.0, 1.0], [1.0, 2.0]],
        columns=["a", "b"],
    )
    result = explain_geo(
        linear_predict,
        background,
        background=background,
        reference=GlobalReference(),
    )
    assert result.shape == (2, 2)
    with pytest.raises(ValueError):
        GeoExplainer(
            linear_predict,
            background,
            reference=GlobalReference(),
        )(background[["b", "a"]])


def test_exact_feature_guard():
    background = np.ones((3, 3))
    with pytest.raises(ValueError):
        GeoExplainer(
            lambda values: values.sum(axis=1),
            background,
            reference=GlobalReference(),
            max_exact_features=2,
        )


def test_spatial_reference_requires_geometry():
    background = np.ones((3, 2))
    explainer = GeoExplainer(
        lambda values: values.sum(axis=1),
        background,
        reference=KernelReference(2.0),
    )
    with pytest.raises(ValueError):
        explainer(background[:1])
