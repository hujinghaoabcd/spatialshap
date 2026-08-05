"""Recover smooth and threshold effects under strong predictor correlation."""

import numpy as np
import pandas as pd

import spatialshap as sshap
from spatialshap import plots

axis = np.linspace(-3.0, 3.0, 41)
background = pd.DataFrame(
    {
        "smooth": axis,
        "threshold": 0.97 * axis + 0.03 * np.sin(5.0 * axis),
    }
)
coordinates = np.column_stack([np.linspace(0.0, 20.0, len(axis)), np.zeros(len(axis))])
rows = [0, 8, 16, 24, 32, 40]
X = background.iloc[rows].copy()
geometry = coordinates[rows]
intercept = 1.0


def smooth(values):
    return 0.8 * np.square(values) - 0.25 * values


def threshold(values):
    return np.where(values >= 0.4, 2.0, -0.7)


def predict(values):
    array = np.asarray(values)
    return intercept + smooth(array[:, 0]) + threshold(array[:, 1])


reference = sshap.KernelReference(bandwidth=3.0, kernel="gaussian")
local_weights = np.vstack(
    [reference.weights(point, coordinates, len(background))[0] for point in geometry]
)
truth = sshap.additive_reference_switch_truth(
    X,
    (smooth, threshold),
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

print("Predictor correlation")
print(background.corr())
print("\nComponent recovery")
print(sshap.geo_recovery_table(result, truth))

fig, ax = plots.recovery_scatter(
    result,
    truth,
    component="geo_interaction",
    feature="threshold",
)
fig.savefig("spatialshap_nonlinear_recovery.png", dpi=180, bbox_inches="tight")
