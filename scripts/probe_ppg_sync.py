"""Probe the format of `ppg_sync/<id>_<camera>_<step>.txt`.

The dataset card says the PPG is sampled at 100 Hz, but the per-line second
column in `ppg_sync.txt` shows inter-sample deltas closer to 1-2 ms (i.e. 500-
1000 Hz). And the PPG-derived HR disagrees with the biomarker `pulse` column
in db.csv by 30+ bpm.

This script enumerates plausible interpretations and prints what each implies
about the sampling rate, total duration, and HR spectrum, so we can pick the
right one before scoring rPPG against this signal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 4.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def _peak_hr(x: np.ndarray, fs: float) -> tuple[float, float]:
    """Return (peak_hr_bpm, peak_power) within the HR band."""
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), 2048))
    band = (f >= 0.7) & (f <= 4.0)
    if not band.any():
        return float("nan"), float("nan")
    f_band, p_band = f[band], pxx[band]
    i = int(np.argmax(p_band))
    return float(f_band[i] * 60.0), float(p_band[i])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/mcd_rppg"))
    parser.add_argument("--patient", default="1020")
    parser.add_argument("--camera", default="FullHDwebcam")
    parser.add_argument("--step", default="before", choices=["before", "after"])
    args = parser.parse_args()

    sync_path = args.root / "ppg_sync" / f"{args.patient}_{args.camera}_{args.step}.txt"
    arr = np.loadtxt(sync_path, dtype=np.float64)
    n = arr.shape[0]
    col1 = arr[:, 0]
    col2 = arr[:, 1]

    print(f"File: {sync_path}")
    print(f"Rows: {n}")
    print()
    print("Column 1 stats (assumed PPG value):")
    print(f"  min={col1.min():.4f}  max={col1.max():.4f}  mean={col1.mean():.4f}  std={col1.std():.4f}")
    print(f"  first 10: {col1[:10].tolist()}")
    print()
    print("Column 2 stats (assumed inter-sample delta or timestamp):")
    print(f"  min={col2.min():.6f}  max={col2.max():.6f}  mean={col2.mean():.6f}  std={col2.std():.6f}")
    print(f"  first 10: {col2[:10].tolist()}")
    print()

    sum_col2 = float(col2.sum())
    monotonic_col2 = bool(np.all(np.diff(col2) >= 0))
    print("--- Interpretations ---")
    print(f"  If col2 == inter-sample delta_s:   sum = {sum_col2:.3f} s   (fs_avg = {n / sum_col2:.1f} Hz)")
    print(f"  If col2 == absolute timestamp_s:   monotonic = {monotonic_col2}, range = {col2.min():.3f}..{col2.max():.3f}")
    print()

    # Pull biomarker pulse for context.
    db = pd.read_csv(args.root / "db.csv")
    row = db[
        (db["patient_id"].astype(str) == args.patient)
        & (db["camera"] == args.camera)
        & (db["step"] == args.step)
    ]
    if len(row):
        bio_pulse = float(row["pulse"].iloc[0])
        print(f"Biomarker pulse (db.csv): {bio_pulse:.1f} bpm")
    else:
        bio_pulse = float("nan")
        print("Biomarker pulse: not found in db.csv")
    print()

    # Try multiple sample-rate assumptions for col1 and report spectral peak.
    print("--- HR estimates from column 1 under different sample-rate assumptions ---")
    print(f"{'fs (Hz)':>10s}  {'duration (s)':>12s}  {'peak HR (bpm)':>14s}  {'peak power':>12s}")
    candidates = [100.0, 500.0, 1000.0]
    # Also try fs derived from col2-as-delta.
    fs_from_deltas = n / sum_col2 if sum_col2 > 0 else None
    if fs_from_deltas is not None:
        candidates.append(round(fs_from_deltas, 1))
    candidates = sorted(set(candidates))
    for fs in candidates:
        if fs <= 1.4:  # need at least 2 * 0.7 Hz
            continue
        try:
            filt = _bandpass(col1 - col1.mean(), fs=fs)
            hr, p = _peak_hr(filt, fs=fs)
            duration = n / fs
            print(f"{fs:>10.1f}  {duration:>12.2f}  {hr:>14.2f}  {p:>12.4e}")
        except Exception as e:
            print(f"{fs:>10.1f}  ERROR: {e}")

    # Also try col2 directly as the PPG signal.
    print()
    print("--- HR estimates assuming col 2 IS the PPG signal ---")
    print(f"{'fs (Hz)':>10s}  {'peak HR (bpm)':>14s}")
    for fs in [100.0, 30.0]:
        try:
            filt = _bandpass(col2 - col2.mean(), fs=fs)
            hr, p = _peak_hr(filt, fs=fs)
            print(f"{fs:>10.1f}  {hr:>14.2f}")
        except Exception as e:
            print(f"{fs:>10.1f}  ERROR: {e}")


if __name__ == "__main__":
    main()
