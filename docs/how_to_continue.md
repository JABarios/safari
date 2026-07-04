# How To Continue SAFARI

This document is the handoff point for resuming work without reconstructing the
whole conversation.

## What Exists Now

Repository:

```text
https://github.com/JABarios/safari
local checkout: /home/juan/pro/safari
```

SAFARI currently has:

- feature extraction from EDF/BDF into per-epoch tabular features;
- training scripts for a centroid baseline and a LightGBM model;
- single-EDF prediction with CSV/NPZ export;
- a local browser app;
- Docker and docker-compose support.

It is separate from human sleep staging. It reuses engineering ideas from Kappa,
but it is an animal research tool.

## Important Local Paths

Local Zenodo rat dataset:

```text
/home/juan/data/data/valenc/external/rat_sleep_zenodo_5227351
```

Local V0 feature/model output from the original development run:

```text
/home/juan/data/data/valenc/pcs_analisis/outputs/rodent_kappa_v0
```

The current trained LightGBM model from that run is:

```text
/home/juan/data/data/valenc/pcs_analisis/outputs/rodent_kappa_v0/rodent_kappa_lgbm_v0.txt
```

When distributing SAFARI, copy or rename that model to:

```text
safari_lgbm_v0.txt
```

Model files are intentionally not committed to git.

## Python Environment

On this machine, use:

```bash
/home/juan/miniconda3/envs/u-sleep/bin/python
```

The system `python3` may be too bare for EDF/MNE/LightGBM work.

## Rebuild The V0 Model

From `/home/juan/pro/safari`:

```bash
/home/juan/miniconda3/envs/u-sleep/bin/python scripts/safari_features.py \
  --source-dir /home/juan/data/data/valenc/external/rat_sleep_zenodo_5227351 \
  --output-dir outputs/safari_v0
```

```bash
/home/juan/miniconda3/envs/u-sleep/bin/python scripts/train_safari_lgbm_v0.py \
  --manifest outputs/safari_v0/feature_manifest.csv \
  --output-dir outputs/safari_v0 \
  --max-epochs-per-record 6000
```

This creates:

```text
outputs/safari_v0/safari_lgbm_v0.txt
```

## Predict One EDF

```bash
/home/juan/miniconda3/envs/u-sleep/bin/python scripts/predict_safari_lgbm_v0.py \
  /path/to/record.edf \
  --model outputs/safari_v0/safari_lgbm_v0.txt \
  --output-csv outputs/safari_v0/predictions/record.csv \
  --output-npz outputs/safari_v0/predictions/record.npz
```

CSV output columns:

```text
epoch,time_s,prediction,confidence,p_wake,p_nrem,p_rem
```

NPZ output contains:

```text
hypnogram, probability, confidence, labels, epoch_s, sfreq, source_edf,
channel_names, channel_map
```

## Run The Local Browser App

Direct Python:

```bash
/home/juan/miniconda3/envs/u-sleep/bin/python scripts/serve_safari.py \
  --data-dir /path/to/edfs \
  --model /path/to/safari_lgbm_v0.txt \
  --output-dir outputs/local_app \
  --host 127.0.0.1 \
  --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

Docker:

```bash
docker build -t safari .
docker run --rm -p 8765:8765 \
  -v /path/to/edfs:/data:ro \
  -v /path/to/model_dir:/models:ro \
  -v /path/to/safari_outputs:/outputs \
  safari
```

The model must be:

```text
/path/to/model_dir/safari_lgbm_v0.txt
```

## What Has Been Verified

Verified locally:

- Python scripts compile.
- Feature extraction runs on Zenodo EDF.
- LightGBM prediction writes CSV/NPZ.
- Direct local web server lists EDFs and stages a record.
- Docker image builds.
- Docker container serves the web app.
- Docker POST prediction writes CSV/NPZ into the mounted output folder.

Smoke EDF used:

```text
AsiagoBleu_180626.edf
```

The smoke prediction matched the existing local model behavior:

```text
Wake 47.2%, NREM 47.4%, REM 5.5%, mean confidence about 0.946
```

## Current Limitations

- No manual correction UI yet.
- No signal viewer yet.
- No channel-map editor yet.
- No explicit postprocessing rules yet.
- Model is trained only on the Zenodo rat dataset.
- Valencia transfer has not been implemented in the SAFARI repo.
- Browser app is intentionally minimal and uses synchronous prediction.
- Model naming still has historical local artifacts in older Valencia outputs;
  public repo naming should use SAFARI.

## Best Next Steps

1. Add a tiny model/artifact release workflow:
   package `safari_lgbm_v0.txt` separately from git.
2. Add explicit channel-map support:
   a small JSON file for cortical/hippocampus/EMG selection.
3. Add Valencia transfer evaluation:
   run SAFARI on `cx/hc/mg` Valencia EDFs and compare with existing `.npz`
   hypnograms and phenotype summaries.
4. Add simple postprocessing:
   median smoothing, minimum bout duration, low-confidence flags.
5. Improve the local web app:
   progress feedback, result table filtering, simple hypnogram plot, manual
   correction export.
6. Start C++ planning:
   freeze feature names and generate a small fixture with expected feature
   values and predictions.

## Collaboration Plan

For a UAM collaborator, do not start with source-code training. Start with:

- Docker installed;
- a folder of EDFs;
- a folder containing `safari_lgbm_v0.txt`;
- the Docker command from `docs/uam_docker_quickstart.md`.

Only after the workflow is useful should we ask them for manual corrections or
new labeled datasets.

