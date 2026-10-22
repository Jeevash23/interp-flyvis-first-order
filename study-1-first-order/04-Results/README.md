# Results - qualification and feasibility

The one-model tutorial reproduction and bounded checks passed. This directory contains actual saved records, figures and the cost projection. No complete interpretability method comparison was executed.

- `qualification/qualification.json`, `native-responses.npz`, `tuning.csv`: 24 native tutorial trajectories, one development model/one speed.
- `qualification/independent-metrics.json`: independent NumPy recomputation.
- `qualification/streaming-validation.json`: native/streaming short controls, source-output identity and locality.
- `qualification/numerical-verification.json`: selected full-trace equality, determinism, precision and timestep diagnostics.
- `qualification/clean-replay.json`: same-machine isolated-environment numerical replay.
- `*-validation.json`: renderer tests. `sparse-central-edge-validation.json` is the latest full-grid central-edge check. Earlier fixtures and interrupted attempts remain retained.
- `cost-projection.json`: arithmetic scenario, not an executed benchmark or confidence interval.
- `figures/`: scientific PNG figures. `pdf-review/`: report rendering QA.

Per-model inference is limited to one model; trial/time/column counts are not independent-model sample sizes. No held-out ensemble data were used for method scoring.
