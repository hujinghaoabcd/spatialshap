# SpatialSHAP

**SpatialSHAP** is a research-oriented Python package for spatially conditioned
Shapley explanations. It makes the counterfactual reference population explicit
and allows that population to vary by focal location.

> **Status — 0.0.1 development:** the repository contains exact conditional and
> joint geographic reference-switch explainers, analytic linear recovery targets,
> component-level validation metrics, and bandwidth sensitivity diagnostics. The
> exact algorithms remain numerical oracles before sampling, tree acceleration,
> and uncertainty inference are introduced.

## Two explanation contracts

### Spatially conditioned feature explanations

For observation \(i\), feature coalition \(S\), and a location-specific reference
measure \(Q_i\), `Explainer` evaluates

\[
v_i(S)=\mathbb{E}_{X_{\bar S}\sim Q_i}
\left[f(x_{i,S},X_{\bar S})\right].
\]

This produces a location-specific baseline and one contribution per feature:

\[
f(x_i)=\phi_{0,i}+\sum_j\phi_{ij}.
\]

### Joint geographic decomposition

`GeoExplainer` adds one grouped `GEO` player. GEO does not enter the predictive
model as an invented coordinate variable. Instead, it switches absent-feature
sampling from a global empirical reference to the focal location's spatial
reference. The result is

\[
f(x_i)=
\phi_0+
\sum_j\phi^{primary}_{ij}+
\phi^{GEO}_i+
\sum_j\phi^{GEO\times j}_{ij}.
\]

The package verifies that sharing every GEO–feature interaction equally between
its two players recovers the ordinary \(p+1\)-player Shapley values.

## Reference distributions

- `GlobalReference` uses one common empirical distribution;
- `KernelReference` uses normalized bisquare, Gaussian, or exponential geographic
  kernel weights;
- `KNNReference` uses the nearest background observations, optionally with inverse
  distance weighting.

Every local reference reports support size, effective sample size, distance range,
and weight concentration.

## Installation

```bash
python -m pip install -e ".[test,plot]"
```

## Conditional quick start

```python
import numpy as np
import pandas as pd
import spatialshap as sshap

rng = np.random.default_rng(42)
background = pd.DataFrame(
    rng.normal(size=(40, 2)),
    columns=["income", "access"],
)
background_coords = rng.uniform(0, 10, size=(40, 2))

def predict(values):
    values = np.asarray(values)
    return 0.5 + 2.0 * values[:, 0] - values[:, 1]

result = sshap.Explainer(
    predict,
    background,
    background_geometry=background_coords,
    reference=sshap.KernelReference(
        bandwidth=3.0,
        kernel="bisquare",
    ),
)(background.iloc[:5], geometry=background_coords[:5])

print(result.summary())
print(result.to_frame())
```

## Joint GEO quick start

```python
geo_result = sshap.GeoExplainer(
    predict,
    background,
    background_geometry=background_coords,
    reference=sshap.KernelReference(
        bandwidth=3.0,
        kernel="gaussian",
    ),
)(background.iloc[:5], geometry=background_coords[:5])

print(geo_result.summary())
print(geo_result.mean_abs_components)

primary = geo_result.primary_values
geo_main = geo_result.geo_values
geo_interactions = geo_result.interaction_values
joint_shapley = geo_result.shapley_values
```

## Analytic recovery validation

For an additive linear prediction function, SpatialSHAP provides the exact
reference-switch truth implied by global and local empirical means.

```python
local_weights = np.vstack(
    [
        reference.weights(point, background_coords, len(background))[0]
        for point in background_coords[:5]
    ]
)
truth = sshap.linear_reference_switch_truth(
    background.iloc[:5],
    np.array([2.0, -1.0]),
    background=background,
    local_weights=local_weights,
    intercept=0.5,
)
report = sshap.geo_recovery_table(geo_result, truth)
print(report)
```

The report contains RMSE, MAE, maximum error, correlation, and sign agreement for
primary, GEO main, GEO interaction, and ordinary joint-Shapley components.

Bandwidth sensitivity is explicit rather than hidden:

```python
profile = sshap.bandwidth_sensitivity(
    predict,
    background.iloc[:5],
    background=background,
    geometry=background_coords[:5],
    background_geometry=background_coords,
    bandwidths=[1.0, 2.0, 5.0, 20.0],
)
```

## Plotting

```python
from spatialshap import plots

fig, ax = plots.bar(result)
fig, ax = plots.beeswarm(result)
fig, ax = plots.waterfall(result[0])
fig, ax = plots.effect_map(result, feature="income")
fig, ax = plots.baseline_map(result)

fig, ax = plots.component_bar(geo_result)
fig, ax = plots.geo_effect_map(geo_result)
fig, ax = plots.interaction_map(geo_result, feature="income")
fig, ax = plots.bandwidth_profile(profile, log_x=True)
```

Plotting functions return Matplotlib objects and never call `plt.show()`.

## Scientific commitments

- independent in-package implementation of coalition evaluation and allocation;
- explicit and auditable global and local reference weights;
- immutable public numerical arrays;
- hard full-coalition efficiency constraint in the joint decomposition;
- per-observation checks of four-component additivity and ordinary Shapley
  equivalence;
- analytic component-level recovery checks, not prediction reconstruction alone;
- no silent coordinate projection, neighbour construction, or missing-value
  imputation;
- unsupported model-output and geometry contracts are rejected explicitly;
- exact algorithms first, so future approximations can be tested against a stable
  numerical oracle.

## Current limitations

- conditional exact enumeration grows as \(2^p\) and defaults to at most 12
  features;
- joint GEO exact enumeration grows as \(2^{p+1}\) and defaults to at most 11
  non-geographic features;
- only single-output numeric prediction is supported;
- geometries are currently numeric coordinate pairs and distance is Euclidean;
- no automatic coordinate transformation or bandwidth selection is performed;
- no uncertainty interval or spatial-block bootstrap is included yet;
- analytic truth currently covers the correctly specified additive linear
  reference-switch game;
- GEO and GEO–feature terms explain a fitted reference-switch game and are not
  automatically causal effects or spatially varying coefficients.

Read `docs/theory.md`, `docs/validation.md`, `docs/limitations.md`, and
`docs/roadmap.md` before treating any output as a scientific spatial effect.
