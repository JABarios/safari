# Model Artifacts

SAFARI code lives in git. Trained models should be distributed separately as
release artifacts, not committed to the repository.

## Current V0 Model

Model id:

```text
safari_lgbm_v0
```

Expected model filename:

```text
safari_lgbm_v0.txt
```

This is the file expected by Docker at:

```text
/models/safari_lgbm_v0.txt
```

## Build A Model Bundle

From the SAFARI repo:

```bash
/home/juan/miniconda3/envs/u-sleep/bin/python scripts/package_safari_model.py \
  --model /home/juan/data/data/valenc/pcs_analisis/outputs/rodent_kappa_v0/rodent_kappa_lgbm_v0.txt \
  --metrics /home/juan/data/data/valenc/pcs_analisis/outputs/rodent_kappa_v0/lgbm_v0_metrics.csv \
  --feature-importance /home/juan/data/data/valenc/pcs_analisis/outputs/rodent_kappa_v0/lgbm_v0_feature_importance.csv \
  --output-dir outputs/model_bundles/safari_lgbm_v0 \
  --zip
```

The bundle contains:

```text
safari_lgbm_v0.txt
manifest.json
MODEL_CARD.md
metrics.csv
feature_importance.csv
```

## Create A GitHub Release

After building the bundle:

```bash
gh release create model-v0.0.1 \
  outputs/model_bundles/safari_lgbm_v0.zip \
  --title "SAFARI model v0.0.1" \
  --notes-file outputs/model_bundles/safari_lgbm_v0/MODEL_CARD.md
```

## Use The Released Model

Download and unzip the release artifact, then run Docker with the unzipped
folder mounted as `/models`:

```bash
docker run --rm -p 8765:8765 \
  -v /path/to/edfs:/data:ro \
  -v /path/to/unzipped_model_bundle:/models:ro \
  -v /path/to/outputs:/outputs \
  safari
```

The file `/models/safari_lgbm_v0.txt` must exist inside the container.

## Why Models Are Not In Git

Model files are generated artifacts. Keeping them out of git lets us:

- version them independently;
- attach model cards and checksums;
- replace or deprecate them cleanly;
- avoid mixing code history with trained artifacts.

