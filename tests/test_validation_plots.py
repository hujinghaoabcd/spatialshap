import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from spatialshap.plots import bandwidth_profile


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
