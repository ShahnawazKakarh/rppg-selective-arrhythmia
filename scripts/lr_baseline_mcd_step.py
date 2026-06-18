"""Logistic-regression baseline for MCD-rPPG resting vs post-exercise.

Reads cached POS pulse waveforms (written by MCDRPPGSegmentDataset),
extracts 4 per-window features, fits LR with the same subject-disjoint
split as the CNN1D-Transformer config, reports per-split macro-F1.

The point: if LR on (HR, HR_std, SNR, template_SQI) lands >0.85, the
pipeline + data carry the class signal and the CNN1D-Transformer's chance
performance is a model/task mismatch (waveform CNN being asked to learn
an FFT internally with ~400 train segments). If LR is also at chance,
the cached pulses don't carry the signal and we have an upstream bug.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rppg_sa.extractors.hr import windowed_hr_bpm
from rppg_sa.extractors.signal_quality import summarize_quality


LABEL_NAMES = ["before", "after"]
LABEL_TO_INDEX = {"before": 0, "after": 1}


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 3.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def _parse_cache_filename(path: Path) -> tuple[str, str] | None:
    """`{patient}_{camera}_{view}_{step}_pos.npy` -> (patient, step). None if malformed."""
    stem = path.stem  # e.g. 1107_FullHDwebcam_front_after_pos
    parts = stem.split("_")
    if len(parts) < 5 or parts[-1] != "pos":
        return None
    patient_id = parts[0]
    step = parts[-2]
    if step not in LABEL_TO_INDEX:
        return None
    return patient_id, step


def _features_for_window(seg: np.ndarray, fs: float) -> np.ndarray:
    hr = windowed_hr_bpm(seg, fs=fs, window_seconds=10.0, overlap=0.5)
    q = summarize_quality(seg, fs=fs)
    return np.array(
        [hr["hr_bpm"], hr["hr_std_bpm"], q["snr_db"], q["template_sqi"]],
        dtype=np.float32,
    )


def build_dataset(
    cache_dir: Path, fs: float, window_seconds: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (X, y, subject_ids) — one row per 30 s segment."""
    win = int(round(fs * window_seconds))
    X: list[np.ndarray] = []
    y: list[int] = []
    subjects: list[str] = []

    cache_files = sorted(cache_dir.glob("*_pos.npy"))
    for cache_path in cache_files:
        parsed = _parse_cache_filename(cache_path)
        if parsed is None:
            continue
        patient_id, step = parsed
        pulse = np.load(cache_path).astype(np.float32)
        if len(pulse) < win:
            continue
        pulse = _bandpass(pulse, fs=fs)

        n_windows = len(pulse) // win
        for i in range(n_windows):
            seg = pulse[i * win : (i + 1) * win]
            feat = _features_for_window(seg, fs=fs)
            if not np.all(np.isfinite(feat)):
                continue
            X.append(feat)
            y.append(LABEL_TO_INDEX[step])
            subjects.append(patient_id)

    return np.stack(X), np.array(y), subjects


def auto_split(subjects: list[str], val_frac: float, test_frac: float, seed: int) -> tuple[set, set, set]:
    unique = sorted(set(subjects))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    return (
        set(unique[n_test + n_val :]),  # train
        set(unique[n_test : n_test + n_val]),  # val
        set(unique[:n_test]),  # test
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/pulse_waveforms"))
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Reading pulse cache from {args.cache_dir}")
    X, y, subjects = build_dataset(args.cache_dir, fs=args.fs, window_seconds=args.window_seconds)
    print(f"  {len(X)} segments | {len(set(subjects))} subjects | classes {np.bincount(y).tolist()}")

    train_s, val_s, test_s = auto_split(subjects, args.val_frac, args.test_frac, args.seed)
    print(f"  subject split: train {len(train_s)} | val {len(val_s)} | test {len(test_s)}")

    train_idx = [i for i, s in enumerate(subjects) if s in train_s]
    val_idx = [i for i, s in enumerate(subjects) if s in val_s]
    test_idx = [i for i, s in enumerate(subjects) if s in test_s]

    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    Xte, yte = X[test_idx], y[test_idx]

    print("\nFeature stats (train):")
    feat_names = ["hr_bpm", "hr_std_bpm", "snr_db", "template_sqi"]
    for i, name in enumerate(feat_names):
        m0 = Xtr[ytr == 0, i].mean()
        m1 = Xtr[ytr == 1, i].mean()
        print(f"  {name:14s}  before={m0:7.3f}  after={m1:7.3f}  Δ={m1 - m0:+.3f}")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=args.C,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    pipe.fit(Xtr, ytr)

    for name, Xs, ys in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        pred = pipe.predict(Xs)
        acc = accuracy_score(ys, pred)
        f1 = f1_score(ys, pred, average="macro")
        print(f"\n[{name}] acc={acc:.3f}  macro-F1={f1:.3f}  (n={len(ys)})")
        print(classification_report(ys, pred, target_names=LABEL_NAMES, digits=3))
        print(f"confusion matrix (rows=true, cols=pred):\n{confusion_matrix(ys, pred)}")

    coefs = pipe.named_steps["lr"].coef_[0]
    print("\nLR coefficients (on standardized features):")
    for name, c in zip(feat_names, coefs):
        print(f"  {name:14s}  {c:+.4f}")


if __name__ == "__main__":
    main()
