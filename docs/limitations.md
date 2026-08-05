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
- the coordinate-group comparison oracle requires a location-aware model whose
  coordinate columns follow the non-geographic features;
- coordinate-group and reference-switch games share a full prediction but generally
  have different empty-coalition baselines and different GEO meanings;
- agreement with the official Kernel GeoShapley implementation validates the
  coordinate-group oracle, not the scientific superiority of either estimand;
- Tree GeoShapley uses a tree-path-dependent value function and is not treated as
  interchangeable with the empirical-background Kernel game;
- the reported four-component basis contains feature primary, GEO main, and
  GEO–feature terms but no explicit feature–feature or higher-order output terms;
- coalition residual and feature-pair second-difference diagnostics reveal when the
  restricted basis is insufficient, but they do not uniquely allocate omitted
  higher-order effects;
- a feature slice of a result is a view for inspection and does not preserve the
  full-prediction additivity identity after omitted features are removed;
- results depend on the fitted model, included features, reference sample, distance,
  kernel, bandwidth or neighbour count, coordinate system, and geographic-player
  definition.

A large reference-switch GEO component does not prove that location causes the
response. It means that changing the counterfactual reference population from
global to local changes the fitted model's explanation. Likewise, a reference-
switch GEO–feature interaction means that the feature contribution changes under
that reference switch; it is not automatically a physical interaction, spatially
varying coefficient, or causal effect.

A coordinate-group GEO component answers another question: how the prediction game
changes when background coordinate inputs are replaced by focal coordinates. It
can be nonzero because of direct coordinate terms or learned location encodings,
even when the local and global feature reference distributions are identical.
Conversely, it can be zero when the model ignores coordinates even though the
reference-switch GEO term is large.

Prediction reconstruction and ordinary joint-Shapley equivalence do not prove that
the four-component basis represents all intermediate coalitions. When structural
residuals or feature-pair second differences are material, feature primary, GEO
main, or GEO–feature terms may be absorbing nonadditive game structure. In that
case the component interpretation should be treated as a restricted projection,
not a complete decomposition of the fitted model.

Users must report the coordinate reference system, distance units, global and local
reference definitions, bandwidth or neighbour count, background sample, model
validation procedure, feature set, geographic-player value function, whether
Kernel or Tree GeoShapley was used in external comparisons, and coalition-
structure diagnostics when interpreting the four-component result.
