"""Cross-check the coordinate-group oracle against geoshapley 0.2.0.0."""

import numpy as np
import pandas as pd
from geoshapley import GeoShapleyExplainer

import spatialshap as sshap

BACKGROUND_FEATURES = pd.DataFrame(
    {
        "density": [-1.5, -0.5, 0.75, 1.5],
        "access": [0.4, -0.8, 1.2, 0.3],
    }
)
BACKGROUND_GEOMETRY = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.5],
        [2.0, 1.5],
        [3.0, 2.0],
    ]
)
ROWS = [0, 2, 3]
X = BACKGROUND_FEATURES.iloc[ROWS].copy()
GEOMETRY = BACKGROUND_GEOMETRY[ROWS]


def predict(values):
    """A location-aware model with feature and GEO-feature structure."""

    array = np.asarray(values, dtype=float)
    density = array[:, 0]
    access = array[:, 1]
    coordinate_x = array[:, 2]
    coordinate_y = array[:, 3]
    return (
        0.7
        + 1.2 * density
        - 0.9 * access
        + 0.5 * coordinate_x
        - 0.3 * coordinate_y
        + 0.4 * density * coordinate_x
        - 0.2 * access * coordinate_y
    )


comparison = sshap.compare_geographic_estimands(
    predict,
    X,
    background=BACKGROUND_FEATURES,
    geometry=GEOMETRY,
    background_geometry=BACKGROUND_GEOMETRY,
    reference=sshap.KernelReference(
        bandwidth=1.5,
        kernel="gaussian",
    ),
)
coordinate = comparison.coordinate_group

background_joint = np.column_stack(
    [BACKGROUND_FEATURES.to_numpy(), BACKGROUND_GEOMETRY]
)
X_joint = pd.DataFrame(
    np.column_stack([X.to_numpy(), GEOMETRY]),
    columns=["density", "access", "coordinate_x", "coordinate_y"],
)
official = GeoShapleyExplainer(
    predict,
    background_joint,
    g=2,
    exact=True,
).explain(X_joint, n_jobs=1)

TOLERANCE = 2e-6
np.testing.assert_allclose(
    coordinate.base_values,
    official.base_value,
    atol=TOLERANCE,
    rtol=TOLERANCE,
)
np.testing.assert_allclose(
    coordinate.primary_values,
    official.primary,
    atol=TOLERANCE,
    rtol=TOLERANCE,
)
np.testing.assert_allclose(
    coordinate.geo_values,
    official.geo,
    atol=TOLERANCE,
    rtol=TOLERANCE,
)
np.testing.assert_allclose(
    coordinate.interaction_values,
    official.geo_intera,
    atol=TOLERANCE,
    rtol=TOLERANCE,
)
np.testing.assert_allclose(
    coordinate.shapley_values,
    official.geoshap_to_shap(),
    atol=TOLERANCE,
    rtol=TOLERANCE,
)

maximum_error = max(
    float(np.max(np.abs(coordinate.primary_values - official.primary))),
    float(np.max(np.abs(coordinate.geo_values - official.geo))),
    float(np.max(np.abs(coordinate.interaction_values - official.geo_intera))),
    float(np.max(np.abs(coordinate.shapley_values - official.geoshap_to_shap()))),
)
print(f"Official GeoShapley 0.2.0.0 maximum component error: {maximum_error:.3e}")
print(comparison.component_table())
