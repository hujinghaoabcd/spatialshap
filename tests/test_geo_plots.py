import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from spatialshap import GeoExplainer, GlobalReference
from spatialshap.plots import component_bar, geo_effect_map, interaction_map


def make_result(with_geometry=True):
    background = pd.DataFrame(
        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        columns=["a", "b"],
    )
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    return GeoExplainer(
        lambda values: values[:, 0] + values[:, 1],
        background,
        reference=GlobalReference(),
        background_geometry=coords if with_geometry else None,
    )(
        background,
        geometry=coords if with_geometry else None,
    )


def test_geo_plots_return_matplotlib_objects():
    result = make_result()
    for plotter, kwargs in [
        (geo_effect_map, {}),
        (interaction_map, {"feature": "a"}),
        (component_bar, {}),
    ]:
        fig, ax = plotter(result, **kwargs)
        assert fig is ax.figure
        fig.clear()


def test_geo_maps_require_geometry_and_validate_feature():
    result = make_result(with_geometry=False)
    with pytest.raises(ValueError, match="requires geometry"):
        geo_effect_map(result)
    result = make_result()
    with pytest.raises(KeyError):
        interaction_map(result, feature="missing")
    with pytest.raises(ValueError, match="positive"):
        component_bar(result, max_display=0)
