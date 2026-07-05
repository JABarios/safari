# Kappa Core Integration

SAFARI should remain the public animal sleep-staging product, but its technical
core should reuse Kappa where possible.

Do not revive or route work through `massai`. That name/repository is discarded.

## Boundary

```text
Kappa
  shared C++/WASM/LightGBM infrastructure
  EDF and signal utilities
  feature-contract discipline
  Python-to-C++ parity tooling

SAFARI
  animal-facing naming and documentation
  rodent/multispecies channel profiles
  Wake/NREM/REM labels
  animal model artifacts
  local browser/Docker workflow
```

SAFARI is not a human PSG tool. It may reuse Kappa internals without sharing
Kappa's human-facing product surface.

## Confirmed Kappa Assets

The private GitHub repository `JABarios/kappa` contains the relevant machinery.
The local `/home/juan/pro/kappa` directory is not a git checkout, but the GitHub
repo is accessible and contains the current source.

Useful files identified in Kappa:

```text
src/analysis/lgbm_infer.h
src/analysis/lgbm_infer.cpp
src/analysis/lgbm_staging.h
src/analysis/lgbm_staging.cpp
src/analysis/lgbm_features_v4.h
src/analysis/lgbm_features_v4.cpp
src/analysis/spectral_metrics.h
src/analysis/spectral_metrics.cpp
src/shared/spectral_utils.h
src/shared/filter_pipeline.h
src/shared/filter_pipeline.cpp
src/shared/hjorth.h
src/shared/MedianFilter.h
src/wasm/kappa_wasm.cpp
tools/lgbm_trainer/
tools/sleep_pipeline_audit/
tests/test_lgbm_infer.cpp
tests/test_lgbm_predict.cpp
tests/test_lgbm_features_v4.cpp
tests/test_lgbm_edf.cpp
```

The `tools/sleep_pipeline_audit` folder is especially important because it
already implements the Python/C++ parity workflow:

```text
dump_python_v4_trace.py
compare_v4_traces.py
compare_v4_signal_traces.py
recompute_python_features_from_cpp_trace.py
```

This matches the desired SAFARI workflow: train/build in Python, then reproduce
features and predictions in C++ with the same exported model.

## First Integration Target

Generalize Kappa's pure C++ LightGBM inference so SAFARI can use the same model
format without Python at runtime.

Current SAFARI model:

```text
safari_lgbm_v0.txt
classes: w, n, r
num_features: 132
```

Kappa `LgbmInfer` currently exposes human-stage assumptions in its public result
shape:

```text
float probs[5]
stageName(int stage)
W, REM, N1, N2, N3
```

Needed change:

```text
dynamic probability vector or configurable max class count
num_classes read from LightGBM model
class labels supplied by SAFARI model manifest
no hardcoded human stage names in the reusable core
```

The reusable layer should be neutral:

```text
LightGBM text model + float feature matrix -> probabilities + argmax
```

SAFARI maps argmax to:

```text
0 = Wake
1 = NREM
2 = REM
```

or whatever class order is recorded in `manifest.json`.

## Feature Porting Plan

Current Python SAFARI feature extractor:

```text
scripts/safari_features.py
```

V0 feature count:

```text
132 features per 4 s epoch
```

Feature families:

```text
cortical band powers
optional hippocampal band powers
EMG band powers
log-ratios
RMS
robust per-record z-scores
previous/next epoch context
```

C++ port should reuse Kappa utilities where possible:

```text
spectral_utils.h
spectral_metrics.*
hjorth.h
MedianFilter.h
filter_pipeline.*
```

But the SAFARI feature contract must stay animal-specific and separately named.
Do not force it into `lgbm_features_v4`, which is human PSG.

Proposed new Kappa-side module:

```text
src/analysis/safari_features.h
src/analysis/safari_features.cpp
tests/test_safari_features.cpp
```

## Parity Tests

Before using C++/WASM for real staging, create fixtures:

```text
one short EDF or synthetic signal
Python feature matrix
Python model probabilities
C++ feature matrix
C++ model probabilities
```

Required checks:

```text
feature_names identical
feature_count identical
max_abs_feature_diff within tolerance
max_abs_probability_diff within tolerance
predicted labels identical or differences explained
```

This should mirror Kappa's existing `sleep_pipeline_audit` approach.

## Suggested Work Order

1. Add `docs/kappa_core_integration.md` to SAFARI. [done]
2. Add a tiny SAFARI Python trace exporter:
   `scripts/dump_safari_trace.py`.
3. In Kappa, generalize `LgbmInfer` away from hardcoded 5-class human results.
4. Add a C++ smoke test that loads `safari_lgbm_v0.txt` and predicts from a
   saved feature row.
5. Port SAFARI feature extraction to C++ using Kappa shared utilities.
6. Add Python/C++ trace comparison for SAFARI.
7. Expose the C++ path to WASM/browser.

## Current Decision

SAFARI can continue to ship the Python/Docker V0 for collaboration and demos,
but new inference/runtime work should be planned around Kappa core reuse.

