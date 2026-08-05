# Roadmap

## Stage 1 — exact conditional core

Implemented:

- global, kernel, and nearest-neighbour references;
- exact single-output Shapley values;
- immutable result object and audit tables;
- summary, local, contribution, baseline, and support maps;
- analytic and property-based numerical validation.

## Stage 2 — joint geographic decomposition

Implemented:

- explicit grouped GEO reference-switch player;
- global-baseline feature primary contributions;
- intrinsic GEO main contribution;
- GEO–feature interaction contributions;
- exact recovery checks against the ordinary joint-game Shapley allocation;
- immutable results, diagnostics, tabular export, maps, examples, and tests.

## Stage 2B — scientific recovery and sensitivity

Implemented:

- analytic four-component truth for additive linear reference-switch games;
- analytic truth for smooth, threshold, and piecewise nonlinear additive models;
- component-level RMSE, MAE, maximum error, correlation, and sign agreement;
- estimated-versus-truth recovery plots;
- global-reference null, dummy-variable, exact-recovery, and correlated-predictor
  additive tests;
- explicit bandwidth sensitivity tables and plotting;
- reproducible linear and nonlinear recovery examples and validation documentation.

Remaining validation work:

- nonadditive feature interactions and higher-order game structure;
- conditional feature-dependence estimands under spatial confounding;
- omitted spatial variables and biased sampling designs;
- reproducible comparison against GeoShapley under matched background contracts;
- kernel, KNN, background-size, and out-of-region transfer experiments;
- manuscript-ready identifiability and estimand discussion.

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
