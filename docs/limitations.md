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
- no joint location-player or location–feature interaction decomposition yet;
- results depend on the model, features, reference sample, distance, kernel, and
  bandwidth.

Users must report the coordinate reference system, distance units, reference
strategy, bandwidth or neighbour count, background sample, and model validation
procedure.
