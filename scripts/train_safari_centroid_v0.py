#!/usr/bin/env python3
"""Train and evaluate the first non-neural SAFARI centroid stager."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_ORDER = np.array(["w", "n", "r"])


def load_feature_files(manifest_path: Path, max_epochs_per_record: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    manifest = pd.read_csv(manifest_path)
    xs = []
    ys = []
    recs = []
    feature_names: list[str] | None = None
    rng = np.random.default_rng(123)
    for _, row in manifest.iterrows():
        data = np.load(row["feature_file"], allow_pickle=True)
        x = data["X"].astype(np.float64)
        y = data["y"].astype(str)
        if feature_names is None:
            feature_names = [str(v) for v in data["feature_names"]]
        if max_epochs_per_record > 0 and len(y) > max_epochs_per_record:
            idx = np.sort(rng.choice(len(y), size=max_epochs_per_record, replace=False))
            x = x[idx]
            y = y[idx]
        xs.append(x)
        ys.append(y)
        recs.append(np.repeat(str(row["record_id"]), len(y)))
    if feature_names is None:
        raise ValueError("No feature files found")
    return np.vstack(xs), np.concatenate(ys), np.concatenate(recs), feature_names


def train_test_records(records: np.ndarray, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    unique = np.array(sorted(set(records)))
    rng = np.random.default_rng(99)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_fraction))) if len(unique) > 1 and test_fraction > 0 else 0
    test_records = set(unique[:n_test])
    test = np.array([rec in test_records for rec in records])
    return ~test, test


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale > 1e-9, scale, np.nanstd(x, axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    return med, scale


def fit_centroids(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, scale = fit_scaler(x)
    z = (x - center) / scale
    centroids = []
    priors = []
    for label in CLASS_ORDER:
        mask = y == label
        if not np.any(mask):
            raise ValueError(f"No examples for class {label}")
        centroids.append(np.nanmean(z[mask], axis=0))
        priors.append(float(np.mean(mask)))
    return center, scale, np.vstack(centroids), np.asarray(priors)


def predict_centroids(x: np.ndarray, center: np.ndarray, scale: np.ndarray, centroids: np.ndarray, priors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = (x - center) / scale
    d2 = np.sum((z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    logits = -0.5 * d2 + np.log(priors[None, :] + 1e-12)
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(axis=1, keepdims=True)
    return CLASS_ORDER[np.argmax(prob, axis=1)], prob


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=int)
    for i, t in enumerate(CLASS_ORDER):
        for j, p in enumerate(CLASS_ORDER):
            out[i, j] = int(np.sum((y_true == t) & (y_pred == p)))
    return out


def cohen_kappa(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    po = np.trace(cm) / n
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float(np.sum(row * col)) / float(n * n)
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else float("nan")


def macro_f1(cm: np.ndarray) -> float:
    vals = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(vals))


def save_model(path: Path, feature_names: list[str], center: np.ndarray, scale: np.ndarray, centroids: np.ndarray, priors: np.ndarray) -> None:
    payload = {
        "model_type": "safari_centroid_v0",
        "classes": CLASS_ORDER.tolist(),
        "feature_names": feature_names,
        "center": center.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "centroids": centroids.astype(float).tolist(),
        "priors": priors.astype(float).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/safari_v0/feature_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/safari_v0"))
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--max-epochs-per-record", type=int, default=6000)
    args = parser.parse_args()

    x, y, records, feature_names = load_feature_files(args.manifest, args.max_epochs_per_record)
    train_mask, test_mask = train_test_records(records, args.test_fraction)
    center, scale, centroids, priors = fit_centroids(x[train_mask], y[train_mask])
    pred_train, _ = predict_centroids(x[train_mask], center, scale, centroids, priors)
    pred_test, _ = predict_centroids(x[test_mask], center, scale, centroids, priors)

    rows = []
    for split, yt, yp in [("train", y[train_mask], pred_train), ("test", y[test_mask], pred_test)]:
        cm = confusion(yt, yp)
        rows.append(
            {
                "split": split,
                "n_epochs": int(cm.sum()),
                "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan"),
                "kappa": cohen_kappa(cm),
                "macro_f1": macro_f1(cm),
            }
        )
        pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_ORDER], columns=[f"pred_{c}" for c in CLASS_ORDER]).to_csv(
            args.output_dir / f"centroid_v0_confusion_{split}.csv"
        )
    metrics = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "centroid_v0_metrics.csv", index=False)
    save_model(args.output_dir / "safari_centroid_v0.json", feature_names, center, scale, centroids, priors)
    print(metrics.to_string(index=False))
    print(args.output_dir / "safari_centroid_v0.json")


if __name__ == "__main__":
    main()
