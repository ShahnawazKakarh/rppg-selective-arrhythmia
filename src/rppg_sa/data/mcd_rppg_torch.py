"""PyTorch Dataset wrapping MCD-rPPG video-derived pulse waveforms.

Mirrors the API of `mitbih_torch.MITBIHSegmentDataset` so `train_classifier.py`
can swap data sources without touching model or training code.

**Task encoded here:** binary `step` classification (resting `before` vs
post-exercise `after`), with subject-disjoint splits keyed on `patient_id`.
This is an end-to-end sanity task before paired AF face-video data is
available; resting/post-exercise is essentially separable by mean HR, so a
working pipeline should hit very high macro-F1.

**Pulse extraction is cached.** POS pulse from face video takes ~16 s per
3-minute clip on M1 Pro. Recomputing per epoch is infeasible. On first
__init__ we extract POS for each unseen record and write a .npy alongside
the dataset's cache root; subsequent runs load directly from cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset

from rppg_sa.data.mcd_rppg import (
    MCDRecord,
    iter_samples,
    load_ppg_sync,
)
from rppg_sa.extractors.face_roi import extract_roi_signals, merge_rois
from rppg_sa.extractors.pos import pos


def _bandpass_and_normalize(
    x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 3.0, order: int = 4
) -> np.ndarray:
    """Bandpass to the HR band and z-score. Matches the v2 baseline."""
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    y = filtfilt(b, a, x)
    mu = float(np.mean(y))
    sd = float(np.std(y)) + 1e-8
    return ((y - mu) / sd).astype(np.float32)


def _segment(pulse: np.ndarray, fs: float, window_seconds: float) -> list[np.ndarray]:
    """Non-overlapping windows. Drops the trailing partial window."""
    win = int(round(fs * window_seconds))
    n = len(pulse) // win
    return [pulse[i * win : (i + 1) * win] for i in range(n)]


class MCDRPPGSegmentDataset(Dataset):
    """Resting / post-exercise classification from POS-extracted rPPG pulse.

    Args:
        root: MCD-rPPG dataset root (the directory containing db.csv).
        cache_dir: Where extracted pulse .npy files are written. Defaults
            to `<root>/../../processed/pulse_waveforms` so the cache lives
            outside `data/raw/`.
        camera: Only records from this camera are used (default
            FullHDwebcam — matches the v2 baseline).
        view: Camera view filter (default "front").
        target_fs: Pulse sampling rate. Defaults to the video frame rate
            of each clip (POS is naturally at video fps); a hard target_fs
            could be added later if the model needs uniform inputs.
        window_seconds: Segment length. 30 s is the standard choice in
            rPPG classification literature and matches MITBIH.
        max_records: Optional cap on records (for fast smoke tests).

    Each item:
        signal: torch.float32 of shape (1, fs * window_seconds).
        label: torch.long, 0 = before (resting), 1 = after (post-exercise).
        record_id: subject patient_id, used by `subject_disjoint_split`.

    NOTE: pulse arrays from different clips can have slightly different
    lengths because video fps varies (most clips are 29.9 Hz, a few are
    24 Hz). We compute window length in samples per-record from that
    record's fs and resample at use time so the tensor shape is uniform.
    """

    LABEL_NAMES = ["before", "after"]  # 0 = resting, 1 = post-exercise
    LABEL_TO_INDEX = {name: i for i, name in enumerate(LABEL_NAMES)}

    def __init__(
        self,
        root: str | Path,
        cache_dir: str | Path | None = None,
        camera: str = "FullHDwebcam",
        view: str = "front",
        target_fs: float = 30.0,
        window_seconds: float = 30.0,
        max_records: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if cache_dir is None:
            # data/raw/mcd_rppg -> data/processed/pulse_waveforms
            cache_dir = self.root.parent.parent / "processed" / "pulse_waveforms"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.view = view
        self.target_fs = float(target_fs)
        self.window_seconds = float(window_seconds)
        self.window_samples = int(round(self.target_fs * self.window_seconds))

        # Filter records by camera/view (and downstream by available files).
        records = [
            r
            for r in iter_samples(self.root)
            if r.camera == camera and r.view == view
        ]
        if max_records is not None:
            records = records[:max_records]

        self.signals: list[np.ndarray] = []
        self.labels: list[int] = []
        self.record_ids: list[str] = []
        self.segment_keys: list[str] = []  # patient_step_winidx for traceability

        n_extracted = 0
        n_cached = 0
        n_skipped = 0
        for rec in records:
            label = self.LABEL_TO_INDEX.get(rec.step)
            if label is None:
                n_skipped += 1
                continue
            try:
                pulse = self._load_or_extract_pulse(rec)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"  skip {rec.patient_id}/{rec.step}: {exc}")
                n_skipped += 1
                continue

            if pulse is None or len(pulse) < self.window_samples:
                n_skipped += 1
                continue

            # Bandpass + normalize the full clip, then segment.
            # (Per-segment normalization would erase relative amplitude, which
            #  matters less for spectral classifiers but more for waveform CNNs.)
            pulse_bn = _bandpass_and_normalize(pulse, fs=self.target_fs)
            for w_idx, seg in enumerate(_segment(pulse_bn, self.target_fs, self.window_seconds)):
                if len(seg) < self.window_samples:
                    continue
                seg = seg[: self.window_samples].astype(np.float32)
                self.signals.append(seg)
                self.labels.append(label)
                self.record_ids.append(rec.patient_id)
                self.segment_keys.append(
                    f"{rec.patient_id}_{rec.step}_{w_idx:02d}"
                )

            if (self.cache_dir / self._cache_name(rec)).exists():
                n_cached += 1
            else:
                n_extracted += 1

        print(
            f"MCDRPPGSegmentDataset: {len(self.signals)} segments from "
            f"{len(set(self.record_ids))} subjects "
            f"(cache hit {n_cached}, extracted {n_extracted}, skipped {n_skipped})"
        )

    # ------------------------------------------------------------------
    # Pulse cache I/O
    # ------------------------------------------------------------------
    def _cache_name(self, rec: MCDRecord) -> str:
        return f"{rec.patient_id}_{self.camera}_{self.view}_{rec.step}_pos.npy"

    def _load_or_extract_pulse(self, rec: MCDRecord) -> np.ndarray:
        """POS pulse waveform, resampled to target_fs.

        Cached as a 1-D float32 array. Sampling rate is stored in the
        accompanying _fs.txt file so a future migration to variable target_fs
        can detect mismatches.
        """
        cache_path = self.cache_dir / self._cache_name(rec)
        fs_path = cache_path.with_suffix(".fs.txt")
        if cache_path.exists() and fs_path.exists():
            pulse = np.load(cache_path)
            fs_cached = float(fs_path.read_text().strip())
            if abs(fs_cached - self.target_fs) < 0.5:
                return pulse  # close enough; videos vary 24-30 Hz
            # Resample if needed.
            return self._resample(pulse, fs_cached, self.target_fs)

        # Extract from video.
        if not rec.video_path.exists():
            raise FileNotFoundError(f"missing video: {rec.video_path}")
        rois = extract_roi_signals(rec.video_path)
        rgb = merge_rois(rois)
        pulse = pos(rgb, fs=rois.fps).astype(np.float32)

        np.save(cache_path, pulse)
        fs_path.write_text(f"{rois.fps:.4f}\n")

        if abs(rois.fps - self.target_fs) >= 0.5:
            pulse = self._resample(pulse, rois.fps, self.target_fs)
        return pulse

    @staticmethod
    def _resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
        from fractions import Fraction

        from scipy.signal import resample_poly

        ratio = Fraction(fs_out).limit_denominator(1000) / Fraction(fs_in).limit_denominator(1000)
        return resample_poly(x, up=ratio.numerator, down=ratio.denominator).astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "signal": torch.from_numpy(self.signals[idx]).unsqueeze(0),  # (1, T)
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "record_id": self.record_ids[idx],
        }

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.LABEL_NAMES}
        for y in self.labels:
            counts[self.LABEL_NAMES[y]] += 1
        return counts


def subject_disjoint_split(
    dataset: MCDRPPGSegmentDataset,
    train_subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
) -> tuple[list[int], list[int], list[int]]:
    """Return indices into `dataset` for subject-disjoint splits.

    Subject-disjoint is essential here: within-subject splits leak the
    person's specific facial pigmentation, lighting, and resting-HR
    baseline into validation.
    """
    sets = {"train": set(train_subjects), "val": set(val_subjects), "test": set(test_subjects)}
    overlap = (
        sets["train"] & sets["val"]
        | sets["train"] & sets["test"]
        | sets["val"] & sets["test"]
    )
    if overlap:
        raise ValueError(f"Subjects appear in multiple splits: {overlap}")

    train_idx, val_idx, test_idx = [], [], []
    for i, sid in enumerate(dataset.record_ids):
        if sid in sets["train"]:
            train_idx.append(i)
        elif sid in sets["val"]:
            val_idx.append(i)
        elif sid in sets["test"]:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


def auto_split_subjects(
    dataset: MCDRPPGSegmentDataset,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministic 70/15/15 subject-level split by patient_id."""
    import random

    subjects = sorted(set(dataset.record_ids))
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = sorted(subjects[:n_test])
    val = sorted(subjects[n_test : n_test + n_val])
    train = sorted(subjects[n_test + n_val :])
    return train, val, test
