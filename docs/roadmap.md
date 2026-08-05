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

## Stage 2C — nonadditive structure diagnostics

Implemented:

- SHAP-kernel weighted intermediate-coalition residual RMSE;
- residual norm relative to the coalition-response scale;
- maximum absolute coalition residual;
- exact non-GEO feature-pair second differences over all valid contexts;
- mean and maximum feature-pair structural burden;
- result-table, summary, map, example, tests, and interpretation documentation;
- explicit separation of prediction reconstruction, ordinary-Shapley equivalence,
  and restricted-basis structural adequacy.

## Stage 2D — GeoShapley estimand comparison

Implemented in the comparison branch:

- exact coordinate-group game matching the value function used by Kernel
  GeoShapley;
- exact location-aware reference-switch game on the same fitted model;
- immutable paired decompositions with a shared full prediction;
- component-level difference metrics and observation-level audit tables;
- estimand scatter plots and spatial difference maps;
- analytic cases separating direct coordinate effects from local compositional
  effects;
- pinned `geoshapley==0.2.0.0` external benchmark on Python 3.13;
- explicit documentation of why the two GEO players answer different questions.

Remaining validation work:

- conditional feature-dependence estimands under spatial confounding;
- omitted spatial variables and biased sampling designs;
- kernel, KNN, background-size, and out-of-region transfer experiments;
- extend the official comparison to Tree GeoShapley without mixing its
  tree-path-dependent value function with the empirical-background Kernel game;
- decide whether a validated richer decomposition should expose explicit
  feature-feature terms or retain them only as diagnostics;
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
