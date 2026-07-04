#!/usr/bin/env python3
"""Predict Wake/NREM/REM for one EDF with a trained SAFARI LightGBM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from safari_features import CLASS_ORDER, extract_features, read_edf_data


def load_model(model_path: Path) -> lgb.Booster:
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return lgb.Booster(model_file=str(model_path))


def check_feature_contract(model: lgb.Booster, feature_names: list[str]) -> None:
    model_names = model.feature_name()
    if not model_names:
        return
    if len(model_names) != len(feature_names):
        raise ValueError(f"Model expects {len(model_names)} features, extractor produced {len(feature_names)}")
    mismatches = [
        (idx, expected, got)
        for idx, (expected, got) in enumerate(zip(model_names, feature_names))
        if expected != got
    ]
    if mismatches:
        preview = "; ".join(f"{idx}: model={exp!r}, extractor={got!r}" for idx, exp, got in mismatches[:5])
        raise ValueError(f"Feature contract mismatch ({len(mismatches)} mismatches): {preview}")


def predict_edf(edf_path: Path, model_path: Path, epoch_s: float) -> dict[str, object]:
    data, sfreq, ch_names, ch_map = read_edf_data(edf_path)
    x, feature_names = extract_features(data, sfreq, ch_map, epoch_s=epoch_s)
    model = load_model(model_path)
    check_feature_contract(model, feature_names)
    prob = np.asarray(model.predict(x), dtype=np.float64)
    if prob.ndim != 2 or prob.shape[1] != len(CLASS_ORDER):
        raise ValueError(f"Expected probability matrix with {len(CLASS_ORDER)} columns, got {prob.shape}")
    pred_idx = np.argmax(prob, axis=1)
    pred = CLASS_ORDER[pred_idx]
    confidence = np.max(prob, axis=1)
    return {
        "prediction": pred,
        "probability": prob,
        "confidence": confidence,
        "feature_names": feature_names,
        "sfreq": sfreq,
        "epoch_s": epoch_s,
        "source_edf": str(edf_path),
        "channel_names": ch_names,
        "channel_map": ch_map.to_json_dict(ch_names),
    }


def write_outputs(result: dict[str, object], csv_path: Path, npz_path: Path | None) -> None:
    pred = np.asarray(result["prediction"]).astype(str)
    prob = np.asarray(result["probability"], dtype=np.float64)
    confidence = np.asarray(result["confidence"], dtype=np.float64)
    epoch_s = float(result["epoch_s"])
    out = pd.DataFrame(
        {
            "epoch": np.arange(1, len(pred) + 1),
            "time_s": np.arange(len(pred)) * epoch_s,
            "prediction": pred,
            "confidence": confidence,
            "p_wake": prob[:, 0],
            "p_nrem": prob[:, 1],
            "p_rem": prob[:, 2],
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    if npz_path is not None:
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            hypnogram=pred.astype("U1"),
            probability=prob.astype(np.float32),
            confidence=confidence.astype(np.float32),
            labels=CLASS_ORDER.astype("U1"),
            epoch_s=float(epoch_s),
            sfreq=float(result["sfreq"]),
            source_edf=str(result["source_edf"]),
            channel_names=np.asarray(result["channel_names"], dtype=object),
            channel_map=json.dumps(result["channel_map"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf", type=Path, help="EDF recording to stage")
    parser.add_argument("--model", type=Path, required=True, help="Trained SAFARI LightGBM text model")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--epoch-s", type=float, default=4.0)
    args = parser.parse_args()

    result = predict_edf(args.edf, args.model, args.epoch_s)
    write_outputs(result, args.output_csv, args.output_npz)
    counts = pd.Series(result["prediction"]).value_counts(normalize=True).reindex(CLASS_ORDER, fill_value=0.0)
    print(args.output_csv)
    if args.output_npz is not None:
        print(args.output_npz)
    print(counts.rename("fraction").to_string())


if __name__ == "__main__":
    main()
