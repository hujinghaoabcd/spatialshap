"""Recover the analytic joint GEO decomposition of a linear model."""

import numpy as np
import pandas as pd

import spatialshap as sshap
from spatialshap import plots

coordinates = np.column_stack([np.linspace(0.0, 10.0, 31), np.zeros(31)])
background = pd.DataFrame(
    {
        "density": 0.5 * coordinates[:, 0] + np.sin(coordinates[:, 0]),
        "access": 2.0 - 0.2 * coordinates[:, 0],
    }
)
coefficients = np.array([1.4, -0.9])
intercept = 3.0


def predict(values):
    return intercept + np.asarray(values) @ coefficients


rows = [0, 8, 15, 22, 30]
X = background.iloc[rows].copy()
geometry = coordinates[rows]
reference = sshap.KernelReference(bandwidth=2.0, kernel="gaussian")
local_weights = np.vstack(
    [
        reference.weights(point, coordinates, len(background))[0]
        for point in geometry
    ]
)

truth = sshap.linear_reference_switch_truth(
    X,
    coefficients,
    background=background,
    local_weights=local_weights,
    intercept=intercept,
)
result = sshap.GeoExplainer(
    predict,
    background,
    reference=reference,
    background_geometry=coordinates,
)(X, geometry=geometry)

print("Component recovery")
print(sshap.geo_recovery_table(result, truth))

profile = sshap.bandwidth_sensitivity(
    predict,
    X,
    background=background,
    geometry=geometry,
    background_geometry=coordinates,
    bandwidths=[0.75, 1.5, 3.0, 10.0, 1000.0],
)
print("\nBandwidth sensitivity")
print(profile)

fig, ax = plots.bandwidth_profile(profile, metric="mean_abs_geo_main", log_x=True)
fig.savefig("spatialshap_bandwidth_profile.png", dpi=180, bbox_inches="tight")
