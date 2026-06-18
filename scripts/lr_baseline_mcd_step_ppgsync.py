"""Same LR baseline as lr_baseline_mcd_step.py, but features are computed
from the contact-PPG ground-truth waveform (ppg_sync col 1) instead of
POS-extracted rPPG.

If POS features land at chance and PPG-sync features land at high F1, the
gap is exactly the rPPG extractor — confirming that POS sub-harmonic lock-on
on high-HR clips erases the class separation and that the eventual selective-
prediction layer (defer when signal quality is low) is the right thesis.
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

from rppg_sa.data.mcd_rppg import iter_samples, load_ppg_sync
from rppg_sa.extractors.hr import windowed_hr_bpm
from rppg_sa.extractors.signal_quality import summarize_quality


LABEL_NAMES = ["before", "after"]
LABEL_TO_INDEX = {"before": 0, "after": 1}


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 3.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def _features_for_window(seg: np.ndarray, fs: float) -> np.ndarray:
    hr = windowed_hr_bpm(seg, fs=fs, window_seconds=10.0, overlap=0.5)
    q = summarize_quality(seg, fs=fs)
    return np.array(
        [hr["hr_bpm"], hr["hr_std_bpm"], q["snr_db"], q["template_sqi"]],
        dtype=np.float32,
    )


def build_dataset(
    root: Path,
    camera: str,
    view: str,
    fs: float,
    window_seconds: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    win = int(round(fs * window_seconds))
    X: list[np.ndarray] = []
    y: list[int] = []
    subjects: list[str] = []

    for rec in iter_samples(root):
        if rec.camera != camera or rec.view != view:
            continue
        if rec.step not in LABEL_TO_INDEX:
            continue
        if not rec.ppg_sync_path.exists():
            continue

        try:
            pulse, _ = load_ppg_sync(rec.ppg_sync_path)
        except Exception:
            continue
        pulse = np.asarray(pulse, dtype=np.float32)
        if len(pulse) < win:
            continue

        pulse = _bandpass(pulse - pulse.mean(), fs=fs)
        n_windows = len(pulse) // win
        for i in range(n_windows):
            seg = pulse[i * win : (i + 1) * win]
            feat = _features_for_window(seg, fs=fs)
            if not np.all(np.isfinite(feat)):
                continue
            X.append(feat)
            y.append(LABEL_TO_INDEX[rec.step])
            subjects.append(rec.patient_id)

    return np.stack(X), np.array(y), subjects


def auto_split(subjects: list[str], val_frac: float, test_frac: float, seed: int):
    unique = sorted(set(subjects))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    return (
        set(unique[n_test + n_val :]),
        set(unique[n_test : n_test + n_val]),
        set(unique[:n_test]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/mcd_rppg"))
    parser.add_argument("--camera", default="FullHDwebcam")
    parser.add_argument("--view", default="front")
    parser.add_argument("--fs", type=float, default=30.0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()

    print(f"Reading PPG-sync from {args.root}")
    X, y, subjects = build_dataset(
        args.root,
        camera=args.camera,
        view=args.view,
        fs=args.fs,
        window_seconds=args.window_seconds,
    )
    print(f"  {len(X)} segments | {len(set(subjects))} subjects | classes {np.bincount(y).tolist()}")

    train_s, val_s, test_s = auto_split(subjects, args.val_frac, args.test_frac, args.seed)
    print(f"  subject split: train {len(train_s)} | val {len(val_s)} | test {len(test_s)}")

    train_idx = [i for i, s in enumerate(subjects) if s in train_s]
    val_idx = [i for i, s in enumerate(subjects) if s in val_s]
    test_idx = [i for i, s in enumerate(subjects) if s in test_s]

    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    Xte, yte = X[test_idx], y[test_idx]

    feat_names = ["hr_bpm", "hr_std_bpm", "snr_db", "template_sqi"]
    print("\nFeature stats (train):")
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
        print(
            f"\n[{name}] acc={accuracy_score(ys, pred):.3f}  "
            f"macro-F1={f1_score(ys, pred, average='macro'):.3f}  (n={len(ys)})"
        )
        print(classification_report(ys, pred, target_names=LABEL_NAMES, digits=3))
        print(f"confusion matrix:\n{confusion_matrix(ys, pred)}")

    coefs = pipe.named_steps["lr"].coef_[0]
    print("\nLR coefficients (standardized):")
    for name, c in zip(feat_names, coefs):
        print(f"  {name:14s}  {c:+.4f}")


if __name__ == "__main__":
    main()
