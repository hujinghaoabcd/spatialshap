"""Basic exact spatially conditioned Shapley example."""

import numpy as np
import pandas as pd

import spatialshap as sshap

rng = np.random.default_rng(42)
background = pd.DataFrame(
    rng.normal(size=(30, 2)),
    columns=["income", "access"],
)
background_geometry = rng.uniform(0, 10, size=(30, 2))


def predict(values):
    values = np.asarray(values)
    return 0.5 + 2.0 * values[:, 0] - values[:, 1]


X = background.iloc[:4].copy()
geometry = background_geometry[:4]

explainer = sshap.Explainer(
    predict,
    background,
    background_geometry=background_geometry,
    reference=sshap.KernelReference(bandwidth=4.0),
)
result = explainer(X, geometry=geometry)
print(result.summary())
print(result.to_frame())
