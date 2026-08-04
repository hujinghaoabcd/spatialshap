# Third-party notices and implementation boundaries

SpatialSHAP is an independent implementation informed by the published Shapley,
SHAP, Joint Shapley, and GeoShapley literature.

The runtime scientific core does not call `shap` or `geoshapley`. Those packages
may be used later in optional reference-comparison tests. No source code is copied
from either project.

Initial theoretical references include:

- Shapley, L. S. (1953), *A Value for n-Person Games*.
- Lundberg, S. M. and Lee, S.-I. (2017), *A Unified Approach to Interpreting Model Predictions*.
- Li, Z. (2024), *GeoShapley: A Game Theory Approach to Measuring Spatial Effects in Machine Learning Models*.

External scientific packages listed in `pyproject.toml` retain their own licences.
