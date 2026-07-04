#!/usr/bin/env python3
"""Train a LightGBM SAFARI V0 stager from cached epoch features."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


CLASS_ORDER = np.array(["w", "n", "r"])
CLASS_TO_ID = {label: idx for idx, label in enumerate(CLASS_ORDER)}


def load_feature_files(manifest_path: Path, max_epochs_per_record: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    manifest = pd.read_csv(manifest_path)
    xs = []
    ys = []
    recs = []
    feature_names: list[str] | None = None
    rng = np.random.default_rng(456)
    for _, row in manifest.iterrows():
        data = np.load(row["feature_file"], allow_pickle=True)
        x = data["X"].astype(np.float32)
        y = data["y"].astype(str)
        if feature_names is None:
            feature_names = [str(v) for v in data["feature_names"]]
        if max_epochs_per_record > 0 and len(y) > max_epochs_per_record:
            idx = np.sort(rng.choice(len(y), size=max_epochs_per_record, replace=False))
            x = x[idx]
            y = y[idx]
        xs.append(x)
        ys.append(np.array([CLASS_TO_ID[v] for v in y], dtype=np.int64))
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


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=int)
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            out[i, j] = int(np.sum((y_true == i) & (y_pred == j)))
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


def evaluate(name: str, y_true: np.ndarray, prob: np.ndarray, output_dir: Path) -> dict[str, float | int | str]:
    pred = np.argmax(prob, axis=1)
    cm = confusion(y_true, pred)
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_ORDER], columns=[f"pred_{c}" for c in CLASS_ORDER]).to_csv(
        output_dir / f"lgbm_v0_confusion_{name}.csv"
    )
    return {
        "split": name,
        "n_epochs": int(cm.sum()),
        "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan"),
        "kappa": cohen_kappa(cm),
        "macro_f1": macro_f1(cm),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/safari_v0/feature_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/safari_v0"))
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--max-epochs-per-record", type=int, default=6000)
    parser.add_argument("--num-boost-round", type=int, default=240)
    args = parser.parse_args()

    x, y, records, feature_names = load_feature_files(args.manifest, args.max_epochs_per_record)
    train_mask, test_mask = train_test_records(records, args.test_fraction)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_data = lgb.Dataset(x[train_mask], label=y[train_mask], feature_name=feature_names, free_raw_data=False)
    test_data = lgb.Dataset(x[test_mask], label=y[test_mask], feature_name=feature_names, reference=train_data, free_raw_data=False)
    params = {
        "objective": "multiclass",
        "num_class": len(CLASS_ORDER),
        "metric": "multi_logloss",
        "learning_rate": 0.045,
        "num_leaves": 31,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 20260704,
        "num_threads": 0,
    }
    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.num_boost_round,
        valid_sets=[test_data],
        valid_names=["test"],
        callbacks=[lgb.log_evaluation(period=40)],
    )

    prob_train = model.predict(x[train_mask])
    prob_test = model.predict(x[test_mask])
    metrics = pd.DataFrame(
        [
            evaluate("train", y[train_mask], prob_train, args.output_dir),
            evaluate("test", y[test_mask], prob_test, args.output_dir),
        ]
    )
    metrics.to_csv(args.output_dir / "lgbm_v0_metrics.csv", index=False)
    model_path = args.output_dir / "safari_lgbm_v0.txt"
    model.save_model(str(model_path))
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "gain": model.feature_importance(importance_type="gain"),
            "split": model.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)
    importance.to_csv(args.output_dir / "lgbm_v0_feature_importance.csv", index=False)
    print(metrics.to_string(index=False))
    print(model_path)


if __name__ == "__main__":
    main()
