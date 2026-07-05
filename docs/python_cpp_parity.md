# Python/C++ Parity

The next serious SAFARI step is to make Kappa C++ reproduce SAFARI Python
predictions. Do this in two stages:

1. C++ loads the LightGBM model and predicts from a Python-exported feature row.
2. C++ computes the SAFARI features itself and matches the Python feature row.

Keeping these separate avoids debugging model inference and feature extraction
at the same time.

## Dump A Python Trace

From the SAFARI repo:

```bash
/home/juan/miniconda3/envs/u-sleep/bin/python scripts/dump_safari_trace.py \
  /path/to/record.edf \
  --model outputs/model_bundles/safari_lgbm_v0/safari_lgbm_v0.txt \
  --epoch 10 \
  --out outputs/parity/record_epoch10_trace.json
```

For small synthetic EDFs, include the whole matrix:

```bash
/home/juan/miniconda3/envs/u-sleep/bin/python scripts/dump_safari_trace.py \
  /path/to/sample.edf \
  --model outputs/model_bundles/safari_lgbm_v0/safari_lgbm_v0.txt \
  --epoch 10 \
  --include-matrix \
  --out outputs/parity/sample_trace.json
```

## Trace Format

The JSON contains:

```text
format
source_edf
model
epoch
epoch_s
sfreq
n_epochs
n_features
classes
channel_names
channel_map
feature_names
feature_row
prediction.stage
prediction.stage_index
prediction.confidence
prediction.probs
```

If `--include-matrix` is used, it also contains:

```text
feature_matrix
predictions
```

## Expected Kappa/C++ Test

First test target:

```text
Load safari_lgbm_v0.txt in Kappa LgbmInfer
Load feature_row from trace JSON
Predict probabilities
Compare against prediction.probs
```

Tolerance target:

```text
max_abs_probability_diff <= 1e-5
same argmax class
```

Second test target:

```text
Load EDF/synthetic signal
Compute SAFARI C++ features
Compare feature_row to Python trace feature_row
```

Tolerance will need to be chosen after seeing FFT/window/resampling numerical
differences.

## Why This Matters

The Python/Docker app is useful for demos and collaboration, but the robust
runtime path is Kappa C++/WASM. This trace file is the bridge between the two.

