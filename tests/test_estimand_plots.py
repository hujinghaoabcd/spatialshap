import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from spatialshap import KNNReference, compare_geographic_estimands
from spatialshap.plots import estimand_difference_map, estimand_scatter


def make_comparison():
    background = pd.DataFrame(
        [[0.0], [1.0], [2.0]],
        columns=["feature"],
    )
    background_geometry = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    )

    def predict(values):
        array = np.asarray(values)
        return 1.0 + 2.0 * array[:, 0] + 3.0 * array[:, 1]

    return compare_geographic_estimands(
        predict,
        background,
        background=background,
        geometry=background_geometry,
        background_geometry=background_geometry,
        reference=KNNReference(k=1),
    )


def test_estimand_plots_return_matplotlib_objects():
    comparison = make_comparison()
    for plotter, kwargs in [
        (estimand_scatter, {}),
        (estimand_scatter, {"component": "baseline"}),
        (
            estimand_scatter,
            {"component": "geo_interaction", "feature": "feature"},
        ),
        (estimand_difference_map, {}),
        (
            estimand_difference_map,
            {"component": "joint_feature_shapley", "feature": 0},
        ),
    ]:
        fig, ax = plotter(comparison, **kwargs)
        assert fig is ax.figure
        fig.clear()


def test_estimand_plots_validate_component_and_feature():
    comparison = make_comparison()
    with pytest.raises(ValueError, match="feature is required"):
        estimand_scatter(comparison, component="primary")
    with pytest.raises(KeyError):
        estimand_scatter(
            comparison,
            component="primary",
            feature="missing",
        )
    with pytest.raises(IndexError):
        estimand_difference_map(
            comparison,
            component="geo_interaction",
            feature=4,
        )
    with pytest.raises(KeyError):
        estimand_scatter(comparison, component="missing")
