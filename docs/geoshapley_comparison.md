# GeoShapley estimand comparison

SpatialSHAP and the original GeoShapley method both use a grouped geographic
player, but the player performs a different operation. Comparing output columns
without matching the cooperative-game value function would therefore be
scientifically misleading.

This chapter defines both games on the same location-aware prediction model,
background observations, focal observations, and exact Shapley decomposition.

## Versioned external reference

The reproducible external check is pinned to `geoshapley==0.2.0.0`. In that
version, `GeoShapleyExplainer` expects non-geographic features followed by `g`
location columns. Those location columns are treated jointly as one player.

The external package uses a large finite kernel weight for the empty and full
coalitions. SpatialSHAP uses the corresponding exact coordinate-group value
function but enforces full-coalition efficiency as a hard equality constraint.
Small floating-point differences are therefore expected even when the estimands
match.

## Coordinate-group game

Let \(x_i\) be focal non-geographic features, \(s_i\) focal coordinates, and
\((X_r,s_r)\) a paired global background row. For feature coalition \(S\) and
location-player indicator \(G\), the coordinate-group game is

\[
v_i^{coord}(S,G)=
\sum_r w_r^{g}
 f\left(
 x_{i,S},
 X_{r,\bar S},
 \begin{cases}
 s_i,&G=1,\\
 s_r,&G=0.
 \end{cases}
 \right).
\]

The geographic player replaces the complete background coordinate block with the
focal coordinate block. This is the grouped-location contract implemented by
Kernel GeoShapley.

Its empty coalition is

\[
v_i^{coord}(\varnothing,0)
=
\sum_r w_r^g f(X_r,s_r),
\]

which is a global model baseline and does not depend on focal location.

## Reference-switch game for a location-aware model

For the reference-switch comparison, focal coordinates remain fixed in every
coalition. The GEO player changes only the measure used to integrate absent
non-geographic features:

\[
v_i^{ref}(S,G)=
\sum_r w_{ir}^{(G)}
 f(x_{i,S},X_{r,\bar S},s_i),
\]

where

\[
w_{ir}^{(0)}=w_r^g,
\qquad
w_{ir}^{(1)}=w_{ir}^{local}.
\]

Its empty coalition is

\[
v_i^{ref}(\varnothing,0)
=
\sum_r w_r^g f(X_r,s_i),
\]

which can vary over focal locations because coordinates are held at \(s_i\).

The full coalition is identical for both games:

\[
v_i^{coord}(N,1)
=
v_i^{ref}(N,1)
=f(x_i,s_i).
\]

Thus both explanations reconstruct the same prediction, but they generally begin
from different baselines and assign different meaning to GEO.

## Shared component basis

Each exact game is projected onto the same component basis:

\[
f(x_i,s_i)=
\phi_{0,i}
+\sum_j\phi^{primary}_{ij}
+\phi_i^{GEO}
+\sum_j\phi^{GEO\times j}_{ij}.
\]

The same structural residual and feature-pair diagnostics are retained for each
game. This separates two questions:

1. does the restricted four-component basis represent each game adequately?;
2. conditional on that adequacy, how much do the two geographic estimands differ?

## Exact comparison API

The prediction callable receives non-geographic features followed by two
coordinate columns:

```python
comparison = spatialshap.compare_geographic_estimands(
    predict_joint,
    X,
    background=background,
    geometry=geometry,
    background_geometry=background_geometry,
    reference=spatialshap.KernelReference(
        bandwidth=2000,
        kernel="gaussian",
    ),
)
```

The result contains two immutable decompositions:

```python
coordinate = comparison.coordinate_group
reference_switch = comparison.reference_switch
```

Both expose:

- `primary_values`;
- `geo_values`;
- `interaction_values`;
- `base_values`;
- `shapley_values`;
- `additivity_error`;
- `decomposition_diagnostics`.

The global comparison table is

```python
print(comparison.component_table())
```

and the observation-level audit table is

```python
frame = comparison.to_frame()
```

All `difference__...` columns are defined as

\[
\text{reference switch}-\text{coordinate group}.
\]

## Visual comparison

A one-to-one scatter plot shows whether the two contracts agree observation by
observation:

```python
fig, ax = spatialshap.plots.estimand_scatter(
    comparison,
    component="geo_main",
)
```

A map identifies where estimand choice changes the result:

```python
fig, ax = spatialshap.plots.estimand_difference_map(
    comparison,
    component="geo_interaction",
    feature="density",
)
```

Supported components are:

- `baseline`;
- `primary`;
- `geo_main`;
- `geo_interaction`;
- `joint_feature_shapley`;
- `joint_geo_shapley`.

Feature-specific components require a feature name or index.

## Diagnostic scenarios

### Coordinate effect without local compositional change

If the fitted model responds directly to coordinates but absent-feature
expectations are the same globally and locally, the coordinate-group GEO term can
be large while the reference-switch GEO term is zero.

### Local compositional change without direct coordinate effect

If the fitted model ignores coordinates but nearby background observations have a
different feature distribution, the coordinate-group GEO term is zero while the
reference-switch GEO and GEO-feature terms can be nonzero.

### Both mechanisms present

When the model uses coordinates and the local feature distribution differs from
the global distribution, both terms can be nonzero. Their difference is not an
implementation error; it reflects different counterfactual questions.

### Spatially correlated features

Both games use empirical background rows and are still interventional coalition
games. Preserving paired background rows does not convert either method into a
fully conditional Shapley estimator under feature dependence.

## Official package cross-check

Install the pinned optional benchmark dependency:

```bash
python -m pip install -e ".[benchmark]"
```

Then run:

```bash
python benchmarks/01_official_geoshapley_v020.py
```

The script compares the SpatialSHAP coordinate-group oracle against the official
`GeoShapleyExplainer` for:

- base value;
- primary effects;
- intrinsic location effect;
- location-feature interactions;
- ordinary Shapley values after half-sharing the interactions.

The repository CI runs this script on Python 3.13. The external package is not a
runtime dependency of SpatialSHAP.

## Reporting recommendations

A comparative paper should report:

1. the exact GeoShapley package version and whether Kernel or Tree mode was used;
2. background sample, weighting and coordinate-column order;
3. the coordinate-group and reference-switch value functions in equations;
4. component-level difference metrics, not only prediction agreement;
5. structural residual diagnostics for both games;
6. maps of estimand differences;
7. sensitivity to local-reference kernel and bandwidth;
8. whether the fitted model directly includes coordinates or location encodings.

## Scientific boundary

The comparison does not declare one geographic estimand universally superior. The
coordinate-group game asks how replacing background location inputs with focal
location changes a prediction game. The reference-switch game asks how replacing
the global counterfactual feature population with a focal spatial population
changes that game. Method choice must follow the scientific question rather than
terminology alone.
