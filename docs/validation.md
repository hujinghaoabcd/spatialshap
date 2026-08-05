# Validation

SpatialSHAP separates numerical correctness from scientific recovery. Additivity
alone is necessary but insufficient: many decompositions can reconstruct the same
prediction. Validation should therefore use data-generating settings in which each
reported component has a known target.

## Analytic linear reference-switch truth

Consider an additive linear prediction function

\[
f(x)=\alpha+\sum_{j=1}^{p}b_jx_j.
\]

Let \(\mu_g\) be the feature mean under the global empirical reference and
\(\mu_i\) the mean under focal location \(i\)'s spatial reference. The joint GEO
game implemented by SpatialSHAP then has the exact representation

\[
\phi^{primary}_{ij}=b_j(x_{ij}-\mu_{g,j}),
\]

\[
\phi^{GEO}_i=\sum_j b_j(\mu_{i,j}-\mu_{g,j}),
\]

\[
\phi^{GEO\times j}_{ij}=-b_j(\mu_{i,j}-\mu_{g,j}).
\]

The global baseline is

\[
\phi_0=\alpha+\sum_j b_j\mu_{g,j}.
\]

These terms exactly reconstruct the prediction. They also recover ordinary joint
game Shapley values after half of every GEO–feature interaction is assigned to
each participating player.

The helper `linear_reference_switch_truth` evaluates this analytic target for any
background matrix, coefficient vector, global weights, and matrix of focal local
weights.

```python
truth = spatialshap.linear_reference_switch_truth(
    X,
    coefficients,
    background=background,
    local_weights=local_weights,
    intercept=intercept,
)
```

## Recovery metrics

`recovery_metrics` reports:

- root mean squared error;
- mean absolute error;
- maximum absolute error;
- Pearson correlation when variation is identifiable;
- sign agreement with an explicit zero tolerance.

`geo_recovery_table` applies the same definitions to feature primary terms, GEO
main effects, GEO–feature interactions, feature joint Shapley values, and the joint
GEO Shapley value.

```python
report = spatialshap.geo_recovery_table(explanation, truth)
```

A manuscript should report component-level recovery rather than only a single
aggregate score. A method can recover total predictions while allocating error to
the wrong component.

## Null and dummy checks

At minimum, simulation studies should include:

1. **Global-reference null:** local and global weights are identical, so GEO main
   and all GEO interactions must be zero.
2. **Dummy variable:** a coefficient is zero even though its spatial distribution
   changes, so its primary and GEO interaction targets remain zero.
3. **Reference convergence:** as a continuous kernel bandwidth becomes very large,
   local weights approach global weights and geographic components should approach
   zero.
4. **Exact recovery:** under a correctly specified additive linear function, all
   four reported components should match their analytic targets to numerical
   precision.

## Bandwidth sensitivity

`bandwidth_sensitivity` reruns the exact joint explainer over an explicit sequence
of bandwidths and returns:

- mean absolute GEO main contribution;
- mean absolute GEO–feature interaction;
- mean absolute joint GEO Shapley value;
- mean effective reference sample size;
- maximum additivity error;
- maximum ordinary-Shapley equivalence error.

```python
profile = spatialshap.bandwidth_sensitivity(
    model,
    X,
    background=background,
    geometry=geometry,
    background_geometry=background_geometry,
    bandwidths=[500, 1000, 2000, 5000],
)

fig, ax = spatialshap.plots.bandwidth_profile(
    profile,
    metric="mean_abs_geo_main",
    log_x=True,
)
```

Bandwidth sensitivity is not a substitute for bandwidth selection. It reveals how
strongly substantive conclusions depend on the geographic reference definition.
The coordinate reference system and distance unit must be reported.

## Interpretation boundary

The analytic target validates the reference-switch cooperative game implemented by
SpatialSHAP. It does not establish that the recovered GEO term is a physical,
causal, or residual spatial effect. Later experiments must separately study model
misspecification, correlated features, omitted spatial variables, spatial sampling
bias, and out-of-region transfer.
