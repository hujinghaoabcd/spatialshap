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
- the reported four-component basis contains feature primary, GEO main, and
  GEO–feature terms but no explicit feature–feature or higher-order output terms;
- coalition residual and feature-pair second-difference diagnostics reveal when the
  restricted basis is insufficient, but they do not uniquely allocate omitted
  higher-order effects;
- a feature slice of a result is a view for inspection and does not preserve the
  full-prediction additivity identity after omitted features are removed;
- results depend on the fitted model, included features, reference sample, distance,
  kernel, bandwidth or neighbour count, and coordinate system.

A large GEO component does not prove that location causes the response. It means
that changing the counterfactual reference population from global to local changes
the fitted model's explanation. Likewise, a GEO–feature interaction means that the
feature contribution changes under that reference switch; it is not automatically
a physical interaction, spatially varying coefficient, or causal effect.

Prediction reconstruction and ordinary joint-Shapley equivalence do not prove that
the four-component basis represents all intermediate coalitions. When structural
residuals or feature-pair second differences are material, feature primary, GEO
main, or GEO–feature terms may be absorbing nonadditive game structure. In that
case the component interpretation should be treated as a restricted projection,
not a complete decomposition of the fitted model.

Users must report the coordinate reference system, distance units, global and local
reference definitions, bandwidth or neighbour count, background sample, model
validation procedure, feature set, whether the conditional or joint geographic
explainer was used, and the coalition-structure diagnostics when interpreting the
four-component result.
