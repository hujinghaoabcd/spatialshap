"""Exact joint geographic reference-switch decomposition."""

import numpy as np
import pandas as pd

import spatialshap as sshap

rng = np.random.default_rng(7)
background = pd.DataFrame(
    {
        "density": rng.normal(0.0, 1.0, 30),
        "access": rng.normal(0.0, 1.0, 30),
    }
)
background_geometry = rng.uniform(0.0, 10.0, size=(30, 2))


def predict(values):
    values = np.asarray(values)
    return 1.0 + 1.8 * values[:, 0] - 0.7 * values[:, 1]


X = background.iloc[:4].copy()
geometry = background_geometry[:4]

result = sshap.GeoExplainer(
    predict,
    background,
    reference=sshap.KernelReference(
        bandwidth=3.5,
        kernel="gaussian",
    ),
    background_geometry=background_geometry,
)(X, geometry=geometry)

print(result.summary())
print(
    result.to_frame()[
        [
            "prediction",
            "base_value",
            "primary__density",
            "geo_main",
            "interaction__GEO__density",
            "additivity_error",
        ]
    ]
)
