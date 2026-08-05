import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from spatialshap import (
    GeoExplainer,
    KernelReference,
    additive_reference_switch_truth,
)
from spatialshap.plots import bandwidth_profile, recovery_scatter


def test_bandwidth_profile_returns_matplotlib_objects():
    profile = pd.DataFrame(
        {
            "bandwidth": [1.0, 2.0, 4.0],
            "mean_abs_geo_main": [2.0, 1.0, 0.5],
        }
    )
    fig, ax = bandwidth_profile(profile)
    assert fig is ax.figure
    assert ax.get_xlabel() == "Bandwidth"
    fig.clear()

    fig, ax = bandwidth_profile(profile, log_x=True)
    assert ax.get_xscale() == "log"
    fig.clear()


def test_bandwidth_profile_validates_columns():
    with pytest.raises(ValueError, match="bandwidth"):
        bandwidth_profile(pd.DataFrame({"value": [1.0]}))
    with pytest.raises(KeyError):
        bandwidth_profile(
            pd.DataFrame({"bandwidth": [1.0]}),
            metric="missing",
        )
    with pytest.raises(ValueError, match="positive"):
        bandwidth_profile(
            pd.DataFrame(
                {
                    "bandwidth": [0.0],
                    "mean_abs_geo_main": [1.0],
                }
            )
        )


def make_recovery_objects():
    background = pd.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0]})
    coordinates = np.column_stack([np.arange(5.0), np.zeros(5)])
    X = background.iloc[[0, 2, 4]].copy()
    geometry = coordinates[[0, 2, 4]]

    def effect(values):
        return np.square(values)

    def predict(values):
        array = np.asarray(values)
        return effect(array[:, 0])

    reference = KernelReference(1.5, kernel="gaussian")
    local_weights = np.vstack(
        [reference.weights(point, coordinates, len(background))[0] for point in geometry]
    )
    truth = additive_reference_switch_truth(
        X,
        (effect,),
        background=background,
        local_weights=local_weights,
    )
    result = GeoExplainer(
        predict,
        background,
        reference=reference,
        background_geometry=coordinates,
    )(X, geometry=geometry)
    return result, truth


def test_recovery_scatter_returns_matplotlib_objects():
    result, truth = make_recovery_objects()
    for component, feature in [
        ("geo_main", None),
        ("joint_geo_shapley", None),
        ("primary", "x"),
        ("geo_interaction", 0),
        ("joint_feature_shapley", "x"),
    ]:
        fig, ax = recovery_scatter(
            result,
            truth,
            component=component,
            feature=feature,
        )
        assert fig is ax.figure
        assert ax.get_xlabel() == "Analytic truth"
        fig.clear()


def test_recovery_scatter_validates_component_and_feature():
    result, truth = make_recovery_objects()
    with pytest.raises(ValueError, match="component"):
        recovery_scatter(result, truth, component="missing")
    with pytest.raises(ValueError, match="feature is required"):
        recovery_scatter(result, truth, component="primary")
    with pytest.raises(KeyError):
        recovery_scatter(result, truth, component="primary", feature="missing")
