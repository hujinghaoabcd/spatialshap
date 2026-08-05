# Third-party notices and implementation boundaries

SpatialSHAP is an independent implementation informed by the published Shapley,
SHAP, Joint Shapley, and GeoShapley literature.

The runtime scientific core does not call `shap` or `geoshapley`. The optional
`benchmark` dependency installs `geoshapley==0.2.0.0` only to cross-check the
independently implemented coordinate-group value function and component outputs.
No GeoShapley source code is copied into SpatialSHAP, and the external package is
not imported by the runtime API.

The GeoShapley package is distributed under the MIT License and retains its own
copyright and licence terms. Its transitive dependencies retain their respective
licences.

Initial theoretical references include:

- Shapley, L. S. (1953), *A Value for n-Person Games*.
- Lundberg, S. M. and Lee, S.-I. (2017), *A Unified Approach to Interpreting Model Predictions*.
- Li, Z. (2024), *GeoShapley: A Game Theory Approach to Measuring Spatial Effects in Machine Learning Models*.

External scientific packages listed in `pyproject.toml` retain their own licences.
