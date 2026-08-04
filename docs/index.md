# SpatialSHAP

SpatialSHAP explains fitted predictions relative to explicit global and spatial
reference populations. The package keeps reference diagnostics visible and offers
two exact development-stage contracts.

## Spatially conditioned feature values

```python
import spatialshap as sshap

result = sshap.Explainer(
    model,
    background,
    background_geometry=background_coords,
    reference=sshap.KernelReference(bandwidth=3000),
)(X, geometry=coords)
```

This contract has a focal local baseline and one value per feature.

## Joint geographic reference-switch decomposition

```python
geo_result = sshap.GeoExplainer(
    model,
    background,
    background_geometry=background_coords,
    reference=sshap.KernelReference(bandwidth=3000),
)(X, geometry=coords)
```

This contract has one global baseline, feature primary effects, an intrinsic GEO
main effect, and one GEO–feature interaction per feature. The GEO player switches
the absent-feature reference from the global population to the focal spatial
population; it is not a coordinate column passed into the model.

Read the theory and limitations before treating any contribution as a scientific
spatial effect.
