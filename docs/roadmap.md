# Roadmap

## Stage 1 — exact conditional core

Implemented:

- global, kernel, and nearest-neighbour references;
- exact single-output Shapley values;
- immutable result object and audit tables;
- summary, local, contribution, baseline, and support maps;
- analytic and property-based numerical validation.

## Stage 2 — joint geographic decomposition

Implemented in the development branch:

- explicit grouped GEO reference-switch player;
- global-baseline feature primary contributions;
- intrinsic GEO main contribution;
- GEO–feature interaction contributions;
- exact recovery checks against the ordinary joint-game Shapley allocation;
- immutable results, diagnostics, tabular export, maps, examples, and tests.

Remaining Stage 2 validation work:

- formal simulation experiments with known data-generating components;
- reproducible comparison against GeoShapley under matched background contracts;
- sensitivity experiments for kernel, bandwidth, neighbour count, and reference
  population;
- manuscript-ready notation and identifiability discussion.

## Stage 3 — inference and scalability

- coalition sampling with deterministic seeds and retained convergence evidence;
- user-supplied spatial-block bootstrap;
- confidence intervals and sign stability;
- batched model calls and cached coalition structures;
- approximation error checks against the exact conditional and joint oracles.

## Stage 4 — richer spatial contracts

- GeoDataFrame and polygon-preserving output;
- fixed and adaptive bandwidth selection experiments;
- road-network and spatiotemporal references only after the planar method is stable.
