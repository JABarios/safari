#!/usr/bin/env python3
"""Dump a SAFARI Python feature/prediction trace for C++ parity tests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from predict_safari_lgbm_v0 import check_feature_contract, load_model
from safari_features import CLASS_ORDER, extract_features, read_edf_data


def to_float_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).ravel()]


def dump_trace(edf_path: Path, model_path: Path, output_path: Path, epoch: int, epoch_s: float, include_matrix: bool) -> None:
    data, sfreq, ch_names, ch_map = read_edf_data(edf_path)
    x, feature_names = extract_features(data, sfreq, ch_map, epoch_s=epoch_s)
    if epoch < 0 or epoch >= x.shape[0]:
        raise ValueError(f"epoch {epoch} outside available range 0..{x.shape[0] - 1}")
    model = load_model(model_path)
    check_feature_contract(model, feature_names)
    prob = np.asarray(model.predict(x), dtype=np.float64)
    pred_idx = np.argmax(prob, axis=1)
    payload: dict[str, object] = {
        "format": "safari_python_trace_v0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_edf": str(edf_path),
        "model": str(model_path),
        "epoch": int(epoch),
        "epoch_s": float(epoch_s),
        "sfreq": float(sfreq),
        "n_epochs": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "classes": CLASS_ORDER.tolist(),
        "channel_names": ch_names,
        "channel_map": ch_map.to_json_dict(ch_names),
        "feature_names": feature_names,
        "feature_row": to_float_list(x[epoch]),
        "prediction": {
            "stage": str(CLASS_ORDER[pred_idx[epoch]]),
            "stage_index": int(pred_idx[epoch]),
            "confidence": float(np.max(prob[epoch])),
            "probs": to_float_list(prob[epoch]),
        },
    }
    if include_matrix:
        payload["feature_matrix"] = x.astype(float).tolist()
        payload["predictions"] = [
            {
                "stage": str(CLASS_ORDER[pred_idx[i]]),
                "stage_index": int(pred_idx[i]),
                "confidence": float(np.max(prob[i])),
                "probs": to_float_list(prob[i]),
            }
            for i in range(x.shape[0])
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edf", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=0, help="0-based epoch index")
    parser.add_argument("--epoch-s", type=float, default=4.0)
    parser.add_argument("--include-matrix", action="store_true", help="Include all epochs/features and predictions")
    args = parser.parse_args()
    dump_trace(args.edf, args.model, args.out, args.epoch, args.epoch_s, args.include_matrix)
    print(args.out)


if __name__ == "__main__":
    main()
