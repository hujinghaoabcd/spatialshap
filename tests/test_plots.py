import matplotlib
import numpy as np

matplotlib.use("Agg")

from spatialshap import Explainer
from spatialshap import plots


def test_initial_plotting_api_returns_matplotlib_objects():
    background = np.array([[0.0, 1.0], [2.0, 3.0], [1.0, 4.0]])
    geometry = np.array([[0.0, 0.0], [1.0, 0.0]])
    explanation = Explainer(
        lambda values: np.asarray(values).sum(axis=1),
        background,
    )(np.array([[1.0, 2.0], [3.0, 4.0]]), geometry=geometry)

    for plotter, kwargs in [
        (plots.bar, {}),
        (plots.beeswarm, {}),
        (plots.effect_map, {"feature": "feature_0"}),
        (plots.baseline_map, {}),
        (plots.effective_reference_map, {}),
    ]:
        fig, ax = plotter(explanation, **kwargs)
        assert fig is ax.figure
        fig.clear()

    fig, ax = plots.waterfall(explanation[0])
    assert fig is ax.figure
    fig.clear()


def test_plot_validation_errors_are_clear():
    import pytest

    background = np.array([[0.0, 1.0], [2.0, 3.0]])
    explanation = Explainer(
        lambda values: np.asarray(values).sum(axis=1),
        background,
    )(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(ValueError, match="one row"):
        plots.waterfall(explanation)
    with pytest.raises(ValueError, match="requires geometry"):
        plots.baseline_map(explanation)
    with pytest.raises(ValueError, match="positive"):
        plots.bar(explanation, max_display=0)
