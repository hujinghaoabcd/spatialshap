# Coalition structure diagnostics

The four-component geographic decomposition is deliberately restricted. For a
joint coalition `A`, it approximates the coalition response relative to the empty
coalition by

\[
v(A)-v(\varnothing)
\approx
\sum_j z_j(A)\alpha_j
+z_G(A)\gamma
+\sum_j z_j(A)z_G(A)\delta_j,
\]

where `z_j` indicates whether feature `j` is present and `z_G` indicates whether
the grouped GEO reference switch is present. The basis contains feature primary
terms, one GEO main term, and GEO–feature interactions. It does not contain
feature–feature or higher-order terms.

Prediction reconstruction and ordinary-Shapley equivalence are therefore not
sufficient evidence that this restricted basis represents the full cooperative
game. SpatialSHAP reports separate structural diagnostics for that purpose.

## SHAP-kernel coalition residual

For every intermediate coalition, define

\[
r(A)=
\left[v(A)-v(\varnothing)\right]
-
\left[
\sum_j z_j(A)\hat\alpha_j
+z_G(A)\hat\gamma
+\sum_jz_j(A)z_G(A)\hat\delta_j
\right].
\]

The decomposition continues to impose full-coalition efficiency as a hard
constraint. The residual diagnostics are calculated over the intermediate
coalitions using the same finite SHAP-kernel weights as the decomposition.

SpatialSHAP reports:

- `weighted_residual_rmse`: weighted root mean squared coalition residual;
- `relative_residual_norm`: weighted residual RMSE divided by the weighted RMS of
  the coalition response relative to the empty coalition;
- `max_abs_coalition_residual`: largest absolute intermediate-coalition residual.

A value close to zero means the restricted four-component basis represents the
sampled game well. A nonzero value does not identify which omitted term is
responsible, but it proves that prediction additivity alone is hiding structural
misfit.

## Feature-pair second differences

For two non-geographic features `j` and `k`, and any context coalition `S`
excluding both, define

\[
\Delta_{jk}v(S)=
v(S\cup\{j,k\})
-v(S\cup\{j\})
-v(S\cup\{k\})
+v(S).
\]

The context may contain GEO and any remaining features. If the game consists only
of feature primary terms, a GEO main term, and GEO–feature interactions, every
feature-pair second difference is zero.

SpatialSHAP reports the mean and maximum absolute second difference over all
non-GEO feature pairs and valid contexts:

- `mean_abs_feature_pair_second_difference`;
- `max_abs_feature_pair_second_difference`.

These quantities detect feature–feature interaction structure, including
higher-order terms that contain a feature pair. They are structural diagnostics,
not a unique allocation of an interaction effect to individual variables.

The exact second differences are also available directly:

```python
second_differences = spatialshap.feature_pair_second_differences(
    coalition_values,
    n_features=3,
)
```

## Result-level audit

Every `GeoExplanation` exposes the diagnostics in the flat result table and in a
compact audit table:

```python
audit = geo_result.diagnostics()
print(
    audit[
        [
            "decomposition_relative_residual_norm",
            "decomposition_max_abs_coalition_residual",
            "decomposition_max_abs_feature_pair_second_difference",
        ]
    ]
)
```

When geometry is available, the structural burden can be mapped:

```python
fig, ax = spatialshap.plots.structure_diagnostic_map(
    geo_result,
    metric="relative_residual_norm",
)
```

Supported map metrics are:

- `weighted_residual_rmse`;
- `relative_residual_norm`;
- `max_abs_coalition_residual`;
- `max_abs_feature_pair_second_difference`.

## Interpretation matrix

### Additivity error small, structural diagnostics small

The final prediction reconstructs and the restricted coalition basis represents
the game well. The four reported components are numerically coherent for the
chosen model and reference contract.

### Additivity error small, structural diagnostics large

The final prediction still reconstructs because efficiency is imposed, but the
four-component basis does not represent the intermediate coalitions. Feature main,
GEO main, or GEO–feature terms may be absorbing omitted feature–feature or
higher-order structure. Substantive interpretation should stop at ordinary joint
Shapley values unless a richer decomposition is explicitly introduced and
validated.

### Feature-pair second differences small, residual large

The omitted structure may involve other forms not summarized by non-GEO pairwise
second differences, numerical instability, or higher-order patterns whose pair
summary is weak after averaging. Inspect the maximum residual, condition number,
and individual simulation design.

### Feature-pair second differences large

The fitted coalition game contains non-GEO pair structure in at least one context.
This is not evidence that the variables interact causally; it is evidence that the
restricted four-component explanation omits coalition structure involving those
features.

## Reporting recommendations

A methods paper should report, by experiment and spatial unit:

1. median, upper quantiles, and maximum relative residual norm;
2. median and maximum absolute feature-pair second difference;
3. maps of both diagnostics when location is substantively relevant;
4. results normalized to an explicit prediction or coalition-response scale;
5. sensitivity to background sample, kernel, bandwidth, and feature set;
6. a known-null additive experiment and a known nonadditive interaction experiment.

There is no universal cutoff that converts these diagnostics into a pass/fail
scientific test. Numerical tolerance can be used for analytic simulations, but
empirical interpretation requires a scale-aware and sensitivity-aware argument.

## Boundary

These diagnostics do not solve conditional Shapley estimation under dependent
features, identify causal interactions, or estimate a unique higher-order
attribution. Their purpose is narrower and essential: they reveal when the current
four-component geographic decomposition is structurally insufficient instead of
allowing omitted interaction structure to remain invisible.
