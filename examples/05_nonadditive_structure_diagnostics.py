"""Diagnose omitted feature-pair structure in the four-component GEO basis."""

import numpy as np
import pandas as pd

import spatialshap as sshap
from spatialshap import plots

axis = np.linspace(-2.5, 2.5, 41)
background = pd.DataFrame(
    {
        "density": axis,
        "access": 0.85 * axis + 0.15 * np.sin(3.0 * axis),
    }
)
coordinates = np.column_stack(
    [
        np.linspace(0.0, 20.0, len(background)),
        np.zeros(len(background)),
    ]
)
rows = [0, 8, 16, 24, 32, 40]
X = background.iloc[rows].copy()
geometry = coordinates[rows]


def predict(values):
    array = np.asarray(values)
    additive = 0.5 + 0.8 * array[:, 0] - 0.4 * array[:, 1]
    feature_pair = 1.2 * array[:, 0] * array[:, 1]
    return additive + feature_pair


result = sshap.GeoExplainer(
    predict,
    background,
    reference=sshap.KernelReference(
        bandwidth=3.5,
        kernel="gaussian",
    ),
    background_geometry=coordinates,
)(X, geometry=geometry)

columns = [
    "decomposition_weighted_residual_rmse",
    "decomposition_relative_residual_norm",
    "decomposition_max_abs_coalition_residual",
    "decomposition_max_abs_feature_pair_second_difference",
    "decomposition_constraint_error",
    "decomposition_shapley_equivalence_error",
]
print(result.diagnostics()[columns])

fig, ax = plots.structure_diagnostic_map(
    result,
    metric="relative_residual_norm",
)
fig.savefig(
    "spatialshap_structure_residual.png",
    dpi=180,
    bbox_inches="tight",
)

fig, ax = plots.structure_diagnostic_map(
    result,
    metric="max_abs_feature_pair_second_difference",
)
fig.savefig(
    "spatialshap_feature_pair_burden.png",
    dpi=180,
    bbox_inches="tight",
)
