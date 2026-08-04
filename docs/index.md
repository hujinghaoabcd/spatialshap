# SpatialSHAP

SpatialSHAP explains a prediction relative to an explicit spatial reference
population. The initial package provides a transparent exact estimator and keeps
all reference-distribution diagnostics visible.

```python
import spatialshap as sshap

result = sshap.Explainer(
    model,
    background,
    background_geometry=background_coords,
    reference=sshap.KernelReference(bandwidth=3000),
)(X, geometry=coords)
```

Read the theory and limitations before treating any contribution as a scientific
spatial effect.
