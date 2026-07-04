#!/usr/bin/env python3
"""Build C++-friendly SAFARI rodent sleep staging features from EDF recordings.

This is the first SAFARI feature contract. It intentionally uses simple
per-epoch time and FFT features that can later be ported to C++/WASM.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd


LABEL_MAP = {0: "w", 1: "n", 2: "r", "0": "w", "1": "n", "2": "r"}
CLASS_ORDER = np.array(["w", "n", "r"])
EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class ChannelMap:
    cortical: list[int]
    hippocampal: list[int]
    emg: list[int]

    def to_json_dict(self, ch_names: list[str]) -> dict[str, list[str]]:
        return {
            "cortical": [ch_names[i] for i in self.cortical],
            "hippocampal": [ch_names[i] for i in self.hippocampal],
            "emg": [ch_names[i] for i in self.emg],
        }


def clean_name(value: object) -> str:
    return str(value).strip().strip("'").strip('"')


def load_zenodo_scoring(scoring_csv: Path) -> dict[str, np.ndarray]:
    raw = pd.read_csv(scoring_csv, header=None, low_memory=False)
    names = [clean_name(x) for x in raw.iloc[1, 1:].tolist()]
    labels = raw.iloc[2:, 1:].reset_index(drop=True)
    out: dict[str, np.ndarray] = {}
    for col_idx, name in enumerate(names):
        vals = labels.iloc[:, col_idx].dropna()
        vals = vals.map(lambda x: LABEL_MAP.get(x, LABEL_MAP.get(str(x), None)))
        vals = vals.dropna().to_numpy(dtype=object)
        out[name] = vals.astype(str)
    return out


def source_scoring_key(edf_path: Path) -> tuple[str, str | None]:
    stem = edf_path.stem
    if stem.endswith("lp"):
        return stem[:-2], "lp"
    if stem.endswith("dp"):
        return stem[:-2], "dp"
    return stem, None


def labels_for_edf(edf_path: Path, scoring: dict[str, np.ndarray], n_epochs: int) -> np.ndarray:
    key, half = source_scoring_key(edf_path)
    if key not in scoring:
        raise KeyError(f"No manual scoring for {edf_path.name} with key {key!r}")
    vals = scoring[key]
    if half == "lp":
        vals = vals[:n_epochs]
    elif half == "dp":
        vals = vals[-n_epochs:]
    else:
        vals = vals[:n_epochs]
    if len(vals) != n_epochs:
        raise ValueError(f"{edf_path.name}: {len(vals)} labels for {n_epochs} epochs")
    return vals


def infer_channel_map(ch_names: list[str]) -> ChannelMap:
    lower = [c.lower() for c in ch_names]
    emg = [i for i, c in enumerate(lower) if "emg" in c or c in {"mg"}]
    unused = {i for i, c in enumerate(lower) if "unused" in c}
    hippocampal = [
        i
        for i, c in enumerate(lower)
        if i not in unused and ("hip" in c or "hc" == c or "hpc" in c)
    ]
    cortical = [
        i
        for i, c in enumerate(lower)
        if i not in unused
        and i not in emg
        and i not in hippocampal
        and (
            "eeg" in c
            or "ecog" in c
            or "cort" in c
            or "front" in c
            or "pariet" in c
            or c in {"cx", "c1", "c2", "c3", "c4", "c5", "c6"}
        )
    ]
    if not cortical:
        cortical = [i for i in range(len(ch_names)) if i not in unused and i not in emg]
    if not cortical:
        raise ValueError(f"Cannot infer cortical channels from {ch_names}")
    return ChannelMap(cortical=cortical, hippocampal=hippocampal, emg=emg)


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale > 1e-9, scale, np.nanstd(x, axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    return np.nan_to_num((x - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def epoch_view(x: np.ndarray, sfreq: float, epoch_s: float) -> np.ndarray:
    epoch_n = int(round(sfreq * epoch_s))
    n_epochs = x.shape[1] // epoch_n
    trimmed = x[:, : n_epochs * epoch_n]
    return trimmed.reshape(x.shape[0], n_epochs, epoch_n)


def bandpower_from_epochs(epochs: np.ndarray, sfreq: float, bands: list[tuple[str, float, float]]) -> tuple[np.ndarray, list[str]]:
    n_ch, n_epochs, epoch_n = epochs.shape
    window = np.hamming(epoch_n).astype(np.float64)
    scale = sfreq * np.sum(window * window)
    freq = np.fft.rfftfreq(epoch_n, d=1.0 / sfreq)
    spec = np.fft.rfft(epochs * window[None, None, :], axis=2)
    psd = (np.abs(spec) ** 2) / max(scale, EPS)
    feats = []
    names = []
    for name, low, high in bands:
        mask = (freq >= low) & (freq < high)
        if not np.any(mask):
            vals = np.zeros((n_ch, n_epochs), dtype=np.float64)
        else:
            vals = np.trapz(psd[:, :, mask], freq[mask], axis=2)
        feats.append(np.log(vals.T + EPS))
        names.extend([f"ch{idx}_{name}_logp" for idx in range(n_ch)])
    return np.column_stack(feats), names


def summarize_family(values: np.ndarray, names: list[str], prefix: str) -> tuple[np.ndarray, list[str]]:
    if values.shape[1] == 1:
        return values, [prefix]
    cols = [
        np.nanmedian(values, axis=1),
        np.nanmax(values, axis=1),
        np.nanmin(values, axis=1),
        np.nanstd(values, axis=1),
    ]
    return np.column_stack(cols), [
        f"{prefix}_median",
        f"{prefix}_max",
        f"{prefix}_min",
        f"{prefix}_std",
    ]


def extract_features(data: np.ndarray, sfreq: float, ch_map: ChannelMap, epoch_s: float = 4.0) -> tuple[np.ndarray, list[str]]:
    epochs = epoch_view(data, sfreq, epoch_s)
    n_ch, n_epochs, _ = epochs.shape
    bands = [
        ("delta", 0.5, 4.0),
        ("theta", 4.0, 8.0),
        ("sigma", 8.0, 15.0),
        ("beta", 15.0, 30.0),
        ("gamma", 30.0, 60.0),
    ]
    all_logp, all_names = bandpower_from_epochs(epochs, sfreq, bands)
    by_name = {name: all_logp[:, i] for i, name in enumerate(all_names)}

    features: list[np.ndarray] = []
    names: list[str] = []

    def add_family(indices: list[int], family: str) -> None:
        if not indices:
            return
        for band, _, _ in bands:
            vals = np.column_stack([by_name[f"ch{i}_{band}_logp"] for i in indices])
            out, out_names = summarize_family(vals, [band], f"{family}_{band}_logp")
            features.append(out)
            names.extend(out_names)

    add_family(ch_map.cortical, "cort")
    add_family(ch_map.hippocampal, "hc")
    add_family(ch_map.emg, "emg")

    cort_delta = np.nanmedian(np.column_stack([by_name[f"ch{i}_delta_logp"] for i in ch_map.cortical]), axis=1)
    cort_theta = np.nanmedian(np.column_stack([by_name[f"ch{i}_theta_logp"] for i in ch_map.cortical]), axis=1)
    cort_sigma = np.nanmedian(np.column_stack([by_name[f"ch{i}_sigma_logp"] for i in ch_map.cortical]), axis=1)
    cort_beta = np.nanmedian(np.column_stack([by_name[f"ch{i}_beta_logp"] for i in ch_map.cortical]), axis=1)
    ratio_cols = [
        cort_theta - cort_delta,
        cort_sigma - cort_delta,
        cort_beta - cort_delta,
    ]
    ratio_names = ["cort_theta_delta_logratio", "cort_sigma_delta_logratio", "cort_beta_delta_logratio"]
    if ch_map.hippocampal:
        hc_theta = np.nanmedian(np.column_stack([by_name[f"ch{i}_theta_logp"] for i in ch_map.hippocampal]), axis=1)
        hc_delta = np.nanmedian(np.column_stack([by_name[f"ch{i}_delta_logp"] for i in ch_map.hippocampal]), axis=1)
        ratio_cols.append(hc_theta - hc_delta)
        ratio_names.append("hc_theta_delta_logratio")
    features.append(np.column_stack(ratio_cols))
    names.extend(ratio_names)

    rms = np.sqrt(np.nanmean(epochs * epochs, axis=2) + EPS).T
    for family, indices in [("cort", ch_map.cortical), ("hc", ch_map.hippocampal), ("emg", ch_map.emg)]:
        if not indices:
            continue
        vals = np.log(rms[:, indices] + EPS)
        out, out_names = summarize_family(vals, ["rms"], f"{family}_rms_log")
        features.append(out)
        names.extend(out_names)

    x = np.column_stack(features)
    z = robust_z(x)
    x = np.column_stack([x, z])
    names = names + [f"{name}_rz" for name in names]

    prev = np.vstack([z[0:1], z[:-1]])
    nxt = np.vstack([z[1:], z[-1:]])
    x = np.column_stack([x, prev, nxt])
    names = names + [f"{name}_prev" for name in names[: z.shape[1]]] + [f"{name}_next" for name in names[: z.shape[1]]]
    if x.shape[0] != n_epochs:
        raise RuntimeError("Feature epoch count mismatch")
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), names


def read_edf_data(edf_path: Path) -> tuple[np.ndarray, float, list[str], ChannelMap]:
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    ch_map = infer_channel_map(ch_names)
    picks = sorted(set(ch_map.cortical + ch_map.hippocampal + ch_map.emg))
    picked_data = raw.get_data(picks=picks).astype(np.float64, copy=False)
    old_to_new = {old: new for new, old in enumerate(picks)}
    picked_map = ChannelMap(
        cortical=[old_to_new[i] for i in ch_map.cortical if i in old_to_new],
        hippocampal=[old_to_new[i] for i in ch_map.hippocampal if i in old_to_new],
        emg=[old_to_new[i] for i in ch_map.emg if i in old_to_new],
    )
    picked_names = [ch_names[i] for i in picks]
    return picked_data, sfreq, picked_names, picked_map


def write_record_features(edf_path: Path, labels: np.ndarray, output_dir: Path, epoch_s: float) -> dict[str, object]:
    data, sfreq, ch_names, ch_map = read_edf_data(edf_path)
    x, feature_names = extract_features(data, sfreq, ch_map, epoch_s=epoch_s)
    n = min(len(labels), x.shape[0])
    x = x[:n]
    labels = labels[:n]
    out_path = output_dir / "features" / f"{edf_path.stem}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=x,
        y=labels.astype("U1"),
        feature_names=np.asarray(feature_names, dtype=object),
        sfreq=float(sfreq),
        epoch_s=float(epoch_s),
        source_edf=str(edf_path),
        channel_names=np.asarray(ch_names, dtype=object),
        channel_map=json.dumps(ch_map.to_json_dict(ch_names)),
    )
    counts = {label: int((labels == label).sum()) for label in CLASS_ORDER}
    return {
        "record_id": edf_path.stem,
        "source_edf": str(edf_path),
        "feature_file": str(out_path),
        "sfreq": sfreq,
        "epoch_s": epoch_s,
        "n_epochs": int(n),
        "n_features": int(x.shape[1]),
        "channel_names": "|".join(ch_names),
        "channel_map": json.dumps(ch_map.to_json_dict(ch_names)),
        **counts,
    }


def build_zenodo_dataset(source_dir: Path, output_dir: Path, limit_records: int, epoch_s: float) -> pd.DataFrame:
    scoring = load_zenodo_scoring(source_dir / "manual_scoring_all_rats.csv")
    edfs = sorted(source_dir.glob("*.edf"))
    if limit_records > 0:
        edfs = edfs[:limit_records]
    rows = []
    for idx, edf_path in enumerate(edfs, start=1):
        data, sfreq, _, _ = read_edf_data(edf_path)
        n_epochs = data.shape[1] // int(round(sfreq * epoch_s))
        labels = labels_for_edf(edf_path, scoring, n_epochs)
        print(f"[{idx}/{len(edfs)}] {edf_path.name}: {n_epochs} epochs", flush=True)
        rows.append(write_record_features(edf_path, labels, output_dir, epoch_s))
    manifest = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "feature_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("/home/juan/data/data/valenc/external/rat_sleep_zenodo_5227351"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/safari_v0"))
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--epoch-s", type=float, default=4.0)
    args = parser.parse_args()
    manifest = build_zenodo_dataset(args.source_dir, args.output_dir, args.limit_records, args.epoch_s)
    print(args.output_dir / "feature_manifest.csv")
    print(manifest[["record_id", "n_epochs", "n_features", "w", "n", "r"]].to_string(index=False))


if __name__ == "__main__":
    main()
