# SAFARI

**Species-Agnostic Framework for Automated sleep scoRIng**

SAFARI is an early research toolkit for non-human animal sleep staging. The
initial target is rodent Wake/NREM/REM scoring from EEG/ECoG/LFP/EMG signals,
with a product path toward C++ and browser/WASM inference.

SAFARI is separate from human sleep staging tools. It reuses engineering ideas
from Kappa where they make sense: tabular features, small models, explicit QC,
portable inference, and review-friendly outputs. It is not a clinical human PSG
tool.

## Current Status

This repository is a V0 seed:

- EDF plus manual scoring ingestion for the Zenodo rat sleep dataset.
- C++-friendly epoch feature extraction.
- A robust centroid baseline exported as JSON.
- A LightGBM tabular model exported as text.
- Grouped train/test evaluation by recording.

No neural-network runtime is required for the deployed V0 models.

## Research-Use Notice

SAFARI is for animal research workflows. It is not intended for medical
diagnosis, human clinical use, or regulatory decision-making.

## Repository Layout

```text
scripts/safari_features.py           EDF -> epoch feature cache
scripts/train_safari_centroid_v0.py  robust centroid baseline
scripts/train_safari_lgbm_v0.py      LightGBM tabular model
scripts/predict_safari_lgbm_v0.py    stage one EDF with a trained model
docs/dataset_registry.csv            public/local dataset planning registry
```

Generated data, EDFs, feature caches, and models are intentionally ignored by
git.

## Environment

The development machine currently uses:

```bash
/home/juan/miniconda3/envs/u-sleep/bin/python
```

Minimum Python dependencies for the current scripts:

```text
numpy
pandas
mne
lightgbm
```

The feature extractor itself is deliberately simple so it can later be ported
to C++/WASM.

## Quick Start

Build features from a local Zenodo 5227351 download:

```bash
python scripts/safari_features.py \
  --source-dir /path/to/rat_sleep_zenodo_5227351 \
  --output-dir outputs/safari_v0
```

Train the centroid baseline:

```bash
python scripts/train_safari_centroid_v0.py \
  --manifest outputs/safari_v0/feature_manifest.csv \
  --output-dir outputs/safari_v0
```

Train the LightGBM model:

```bash
python scripts/train_safari_lgbm_v0.py \
  --manifest outputs/safari_v0/feature_manifest.csv \
  --output-dir outputs/safari_v0
```

Stage a new EDF with the LightGBM model:

```bash
python scripts/predict_safari_lgbm_v0.py /path/to/record.edf \
  --model outputs/safari_v0/safari_lgbm_v0.txt \
  --output-csv outputs/safari_v0/predictions/record.csv \
  --output-npz outputs/safari_v0/predictions/record.npz
```

## First Local Benchmark

On the local Zenodo rat dataset cache, using 44 EDF recordings and a held-out
recording split with at most 6000 epochs per record:

| model | split | epochs | accuracy | kappa | macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| centroid_v0 | test | 66000 | 0.8528 | 0.7546 | 0.7829 |
| lgbm_v0 | test | 66000 | 0.9292 | 0.8742 | 0.9040 |

These are early internal numbers, not a validation claim.

## Design Direction

SAFARI should support capability profiles rather than hardcoded datasets:

- cortex + EMG;
- cortex + hippocampus + EMG;
- multi-cortical + EMG;
- multi-cortical + hippocampus + EMG.

The browser version should eventually support review/correction, versioned
models, channel maps, confidence/QC flags, and exportable annotations.
