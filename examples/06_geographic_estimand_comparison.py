"""Compare coordinate-group GeoShapley with the reference-switch game."""

import numpy as np
import pandas as pd

import spatialshap as sshap
from spatialshap import plots

coordinate_x = np.linspace(0.0, 20.0, 41)
coordinate_y = 2.0 * np.sin(coordinate_x / 4.0)
background_geometry = np.column_stack([coordinate_x, coordinate_y])
background = pd.DataFrame(
    {
        "density": 0.18 * coordinate_x + np.sin(coordinate_x / 2.5),
        "access": np.cos(coordinate_x / 3.0) - 0.04 * coordinate_x,
    }
)
rows = [0, 8, 16, 24, 32, 40]
X = background.iloc[rows].copy()
geometry = background_geometry[rows]


def predict(values):
    array = np.asarray(values)
    density = array[:, 0]
    access = array[:, 1]
    x_coord = array[:, 2]
    y_coord = array[:, 3]
    return (
        1.0
        + 0.9 * density
        - 0.7 * access
        + 0.08 * x_coord
        - 0.12 * y_coord
        + 0.05 * density * x_coord
    )


comparison = sshap.compare_geographic_estimands(
    predict,
    X,
    background=background,
    geometry=geometry,
    background_geometry=background_geometry,
    reference=sshap.KernelReference(
        bandwidth=4.0,
        kernel="gaussian",
    ),
)

print(comparison.summary())
print("\nObservation-level output")
print(comparison.to_frame())

fig, ax = plots.estimand_scatter(
    comparison,
    component="geo_main",
)
fig.savefig(
    "spatialshap_estimand_geo_scatter.png",
    dpi=180,
    bbox_inches="tight",
)

fig, ax = plots.estimand_difference_map(
    comparison,
    component="geo_interaction",
    feature="density",
)
fig.savefig(
    "spatialshap_estimand_interaction_difference.png",
    dpi=180,
    bbox_inches="tight",
)
