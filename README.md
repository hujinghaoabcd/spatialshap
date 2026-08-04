# SpatialSHAP

**SpatialSHAP** is a research-oriented Python package for spatially conditioned
Shapley explanations. It changes the reference distribution used by a Shapley
value calculation so that each focal location can be explained relative to a
geographically meaningful background.

> **Status — 0.0.1 development scaffold:** global, kernel-weighted, and nearest-
> neighbour references are implemented together with exact single-output
> conditional Shapley values, immutable result objects, tabular export, spatial
> diagnostics, and initial Matplotlib plots. Joint location-player and
> location–feature interaction decomposition are the next method stage.

## Scientific scope

For observation \(i\), feature coalition \(S\), and a location-specific reference
measure \(Q_i\), SpatialSHAP evaluates

\[
v_i(S)=\mathbb{E}_{X_{\bar S}\sim Q_i}
\left[f(x_{i,S},X_{\bar S})\right].
\]

The package keeps the Shapley allocation rule unchanged. Its first methodological
contribution is the explicit construction and auditing of \(Q_i\):

- `GlobalReference` uses one common empirical distribution;
- `KernelReference` uses normalized geographic kernel weights;
- `KNNReference` uses the nearest background observations.

The current exact estimator is intentionally restricted to modest feature counts.
It is designed as a transparent numerical oracle before sampling and tree-specific
accelerators are introduced.

## Installation

```bash
python -m pip install -e ".[test,plot]"
```

## Quick start

```python
import numpy as np
import pandas as pd
import spatialshap as sshap

rng = np.random.default_rng(42)
X_train = pd.DataFrame(
    rng.normal(size=(40, 2)),
    columns=["income", "access"],
)
coords_train = rng.uniform(0, 10, size=(40, 2))

coef = np.array([2.0, -1.0])

def predict(values):
    return np.asarray(values) @ coef + 0.5

X_test = X_train.iloc[:5].copy()
coords_test = coords_train[:5]

explainer = sshap.Explainer(
    predict,
    background=X_train,
    background_geometry=coords_train,
    reference=sshap.KernelReference(
        bandwidth=3.0,
        kernel="bisquare",
    ),
)

explanation = explainer(X_test, geometry=coords_test)
print(explanation.summary())
print(explanation.to_frame())
```

## Plotting

```python
from spatialshap import plots

fig, ax = plots.bar(explanation)
fig, ax = plots.beeswarm(explanation)
fig, ax = plots.waterfall(explanation[0])
fig, ax = plots.effect_map(explanation, feature="income")
fig, ax = plots.baseline_map(explanation)
```

Plotting functions return Matplotlib objects and never call `plt.show()`.

## Initial commitments

- independent in-package implementation of coalition evaluation and Shapley sums;
- explicit reference weights and diagnostics;
- immutable public numerical arrays;
- no silent coordinate projection, neighbour construction, or missing-value imputation;
- single-output regression first, with unsupported contracts rejected explicitly;
- global-reference tests against analytic linear-model results;
- local-reference additivity tests at every observation;
- a small public API rather than a large plugin framework.

## Current limitations

- exact enumeration grows exponentially and defaults to at most 12 features;
- only single-output numeric prediction is supported;
- geometries are currently represented by numeric coordinate pairs;
- the current method explains prediction relative to a spatial reference and does
  not estimate causal effects;
- joint location-player GeoShapley decomposition is not yet included.

See `docs/theory.md`, `docs/limitations.md`, and `docs/roadmap.md`.
