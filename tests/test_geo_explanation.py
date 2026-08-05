import numpy as np
import pandas as pd
import pytest

from spatialshap import GeoExplainer, GlobalReference


def make_result():
    background = pd.DataFrame(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        columns=["a", "b"],
    )
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    return GeoExplainer(
        lambda values: values[:, 0] + 2 * values[:, 1],
        background,
        reference=GlobalReference(),
        background_geometry=coords,
    )(background, geometry=coords)


def test_result_arrays_are_read_only():
    result = make_result()
    with pytest.raises(ValueError):
        result.primary_values[0, 0] = 99
    with pytest.raises(ValueError):
        result.shapley_values[0, 0] = 99


def test_result_slicing_and_frame():
    result = make_result()
    sliced = result[:2, "a"]
    assert sliced.shape == (2, 1)
    assert sliced.feature_names == ("a",)
    frame = result.to_frame()
    expected = {
        "geo_main",
        "primary__a",
        "interaction__GEO__a",
        "shapley__a",
        "shapley__GEO",
        "decomposition_weighted_residual_rmse",
        "decomposition_relative_residual_norm",
        "decomposition_max_abs_feature_pair_second_difference",
        "decomposition_constraint_error",
    }
    assert expected.issubset(frame.columns)
    diagnostics = result.diagnostics()
    assert diagnostics.shape == (3, 12)
    assert np.max(np.abs(diagnostics["decomposition_relative_residual_norm"])) < 1e-12


def test_summary_and_component_table():
    result = make_result()
    summary = result.summary()
    assert "GeoExplanation" in summary
    assert "max_relative_structure_residual" in summary
    assert "max_feature_pair_second_difference" in summary
    assert list(result.mean_abs_components.index) == ["b", "a"]
    assert result.shapley_feature_names == ("a", "b", "GEO")
