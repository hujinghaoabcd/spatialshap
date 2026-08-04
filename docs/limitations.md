# Limitations

SpatialSHAP attributes a fitted model's prediction relative to a chosen spatial
reference distribution. It does not by itself identify a causal effect or a true
physical spatial process.

Current limitations include:

- single-output numeric predictions only;
- exact exponential-time enumeration;
- numeric coordinate pairs rather than general geometry objects in the core;
- Euclidean distance without automatic CRS transformation;
- no uncertainty interval in the initial implementation;
- the grouped GEO player is defined specifically as a switch from global to focal
  spatial reference weights, not as a raw coordinate feature;
- GEO main and GEO–feature interaction terms depend on the restricted decomposition
  basis and can absorb higher-order game structure;
- a feature slice of a result is a view for inspection and does not preserve the
  full-prediction additivity identity after omitted features are removed;
- results depend on the fitted model, included features, reference sample, distance,
  kernel, bandwidth or neighbour count, and coordinate system.

A large GEO component does not prove that location causes the response. It means
that changing the counterfactual reference population from global to local changes
the fitted model's explanation. Likewise, a GEO–feature interaction means that the
feature contribution changes under that reference switch; it is not automatically
a physical interaction, spatially varying coefficient, or causal effect.

Users must report the coordinate reference system, distance units, global and local
reference definitions, bandwidth or neighbour count, background sample, model
validation procedure, feature set, and whether the conditional or joint geographic
explainer was used.
