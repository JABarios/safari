#!/usr/bin/env python3
"""Package a trained SAFARI model as a release-ready artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_exists(src: Path | None, dst: Path) -> str | None:
    if src is None or not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.name


def write_model_card(path: Path, manifest: dict[str, object]) -> None:
    metrics = manifest.get("metrics", {})
    test = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    path.write_text(
        f"""# SAFARI LightGBM V0 Model Card

## Identity

- Model id: `{manifest["model_id"]}`
- File: `{manifest["model_file"]}`
- SHA256: `{manifest["model_sha256"]}`
- Created UTC: `{manifest["created_utc"]}`

## Intended Use

This model is an early SAFARI Wake/NREM/REM scorer for non-human animal sleep
research. It is intended for exploratory rodent sleep staging workflows and
local review, not for human clinical sleep scoring or medical decision-making.

## Training Data

The V0 model was trained from the local copy of Zenodo record 5227351, using EDF
signals and manual 4 s Wake/NREM/REM scoring.

Source:

```text
https://zenodo.org/records/5227351
```

## Inputs

The current feature extractor infers cortical, optional hippocampal, and EMG
channels from EDF/BDF channel names. Epoch length defaults to 4 s.

## Outputs

For each epoch:

- predicted label: `w`, `n`, or `r`;
- per-class probabilities: Wake, NREM, REM;
- confidence: maximum class probability.

## Internal Benchmark

Held-out recording split, sampled at up to 6000 epochs per record:

```text
test accuracy: {test.get("accuracy", "NA")}
test kappa:    {test.get("kappa", "NA")}
test macro F1: {test.get("macro_f1", "NA")}
```

These are early internal numbers, not a validation claim.

## Known Limitations

- Trained only on the Zenodo rat dataset.
- Not yet validated on Valencia PCS/BDL/HA or future UAM recordings.
- No manual correction loop yet.
- No explicit channel-map editor yet.
- No postprocessing beyond model probabilities.

## Research-Use Notice

SAFARI is an animal research tool. It is not intended for human clinical sleep
scoring or medical decision-making.
""",
        encoding="utf-8",
    )


def load_metrics(metrics_csv: Path | None) -> dict[str, dict[str, float | int | str]]:
    if metrics_csv is None or not metrics_csv.exists():
        return {}
    df = pd.read_csv(metrics_csv)
    out: dict[str, dict[str, float | int | str]] = {}
    for row in df.to_dict(orient="records"):
        split = str(row.pop("split"))
        out[split] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trained LightGBM text model")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_bundles/safari_lgbm_v0"))
    parser.add_argument("--model-id", default="safari_lgbm_v0")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--feature-importance", type=Path, default=None)
    parser.add_argument("--zip", action="store_true", help="Also write a .zip next to the bundle directory")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.output_dir / f"{args.model_id}.txt"
    shutil.copy2(args.model, model_out)
    model = lgb.Booster(model_file=str(model_out))
    metrics_file = copy_if_exists(args.metrics, args.output_dir / "metrics.csv")
    importance_file = copy_if_exists(args.feature_importance, args.output_dir / "feature_importance.csv")
    manifest = {
        "model_id": args.model_id,
        "model_type": "lightgbm_multiclass",
        "classes": ["w", "n", "r"],
        "model_file": model_out.name,
        "model_sha256": sha256_file(model_out),
        "model_size_bytes": model_out.stat().st_size,
        "num_features": len(model.feature_name()),
        "feature_names": model.feature_name(),
        "metrics_file": metrics_file,
        "feature_importance_file": importance_file,
        "metrics": load_metrics(args.metrics),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_use_only": True,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_model_card(args.output_dir / "MODEL_CARD.md", manifest)
    if args.zip:
        zip_path = args.output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(args.output_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(args.output_dir.parent))
        print(zip_path)
    print(args.output_dir)
    print(manifest_path)
    print(f"sha256 {manifest['model_sha256']}")


if __name__ == "__main__":
    main()
