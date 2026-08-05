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

## Nonlinear additive truth

The same derivation extends to

\[
f(x)=\alpha+\sum_{j=1}^{p}g_j(x_j),
\]

where each \(g_j\) can be smooth, thresholded, or piecewise. Let

\[
m_{g,j}=E_g[g_j(X_j)],
\qquad
m_{i,j}=E_i[g_j(X_j)].
\]

Then

\[
\phi^{primary}_{ij}=g_j(x_{ij})-m_{g,j},
\]

\[
\phi^{GEO}_i=\sum_j(m_{i,j}-m_{g,j}),
\]

\[
\phi^{GEO\times j}_{ij}=-(m_{i,j}-m_{g,j}).
\]

`additive_reference_switch_truth` evaluates these targets from a sequence of
one-dimensional feature functions.

```python
truth = spatialshap.additive_reference_switch_truth(
    X,
    (lambda x: x**2, lambda x: np.where(x > 0, 1.0, -1.0)),
    background=background,
    local_weights=local_weights,
    intercept=intercept,
)
```

This permits exact tests with strongly correlated predictors because the fitted
response remains additive. It does **not** resolve the broader distinction between
interventional and conditional Shapley games under feature dependence.

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

`plots.recovery_scatter` gives an observation-level estimated-versus-truth check:

```python
fig, ax = spatialshap.plots.recovery_scatter(
    explanation,
    truth,
    component="geo_interaction",
    feature="population_density",
)
```

## Null and dummy checks

At minimum, simulation studies should include:

1. **Global-reference null:** local and global weights are identical, so GEO main
   and all GEO interactions must be zero.
2. **Dummy variable:** a coefficient or feature function is zero even though its
   spatial distribution changes, so its primary and GEO interaction targets remain
   zero.
3. **Reference convergence:** as a continuous kernel bandwidth becomes very large,
   local weights approach global weights and geographic components should approach
   zero.
4. **Linear exact recovery:** under a correctly specified additive linear function,
   all four components should match analytic targets.
5. **Nonlinear additive exact recovery:** smooth and threshold functions should also
   match analytic targets when the prediction function is additive.

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

The analytic targets validate the reference-switch cooperative game implemented by
SpatialSHAP. They do not establish that the recovered GEO term is a physical,
causal, or residual spatial effect. Later experiments must separately study model
misspecification, nonadditive feature interactions, conditional feature dependence,
omitted spatial variables, spatial sampling bias, and out-of-region transfer.
