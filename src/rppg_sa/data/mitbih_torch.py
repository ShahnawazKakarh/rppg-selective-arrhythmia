"""PyTorch Dataset wrapping MIT-BIH 30-second segments.

Used to validate the full classifier + UQ + selective stack on fully-open data
before paired rPPG + AF face-video data is available. The same Dataset API is
used downstream for rPPG-derived pulse waveforms — only the source differs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from rppg_sa.data.mitbih import CLASS_TO_INDEX, load_dataset


def _bandpass_and_normalize(x: np.ndarray, fs_in: float, fs_out: float = 128.0) -> np.ndarray:
    """Resample to `fs_out` Hz and z-score normalize.

    Resampling makes the model input rate independent of the source modality
    (MIT-BIH at 360 Hz vs. rPPG at ~30 Hz). 128 Hz is comfortably above the
    physiological pulse harmonics we care about.
    """
    if fs_in != fs_out:
        # Use rational resampling for precision.
        # 360 -> 128: up=32, down=90 (approx via gcd reduction handled by Fraction).
        from fractions import Fraction

        ratio = Fraction(fs_out).limit_denominator(1000) / Fraction(fs_in).limit_denominator(1000)
        x = resample_poly(x, up=ratio.numerator, down=ratio.denominator)
    mu = x.mean()
    sd = x.std() + 1e-8
    return ((x - mu) / sd).astype(np.float32)


class MITBIHSegmentDataset(Dataset):
    """30-second ECG windows from MIT-BIH, labelled NSR / AF / Other.

    Args:
        root: Directory containing MIT-BIH records (with .hea and .dat files).
        record_ids: Optional subset of record IDs.
        target_fs: Target sampling rate after resampling (Hz).
        window_seconds: Segment length used during dataset construction.

    Each item:
        signal: torch.float32 tensor of shape (1, target_fs * window_seconds).
        label: torch.long tensor with value in {0: NSR, 1: AF, 2: Other}.
    """

    LABEL_NAMES = ["NSR", "AF", "Other"]

    def __init__(
        self,
        root: str | Path,
        record_ids: list[str] | None = None,
        target_fs: float = 128.0,
        window_seconds: float = 30.0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.target_fs = target_fs
        self.window_seconds = window_seconds
        segments = load_dataset(
            self.root,
            record_ids=record_ids,
            window_seconds=window_seconds,
            step_seconds=window_seconds,
        )
        if not segments:
            raise FileNotFoundError(
                f"No MIT-BIH segments found under {self.root}. "
                "Run scripts/download_mitbih.py first."
            )
        # Process and cache in memory; full MIT-BIH at 128 Hz is < 200 MB float32.
        self.signals: list[np.ndarray] = []
        self.labels: list[int] = []
        self.record_ids: list[str] = []
        for seg in segments:
            sig = _bandpass_and_normalize(seg["signal"], fs_in=seg["fs"], fs_out=target_fs)
            expected_len = int(target_fs * window_seconds)
            if len(sig) < expected_len:
                continue
            sig = sig[:expected_len]
            self.signals.append(sig)
            self.labels.append(seg["label_idx"])
            self.record_ids.append(seg["record_id"])

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
    dataset: MITBIHSegmentDataset,
    train_records: list[str],
    val_records: list[str],
    test_records: list[str],
) -> tuple[list[int], list[int], list[int]]:
    """Return indices into `dataset` for subject-disjoint train/val/test splits.

    Subject-disjoint (record-disjoint, in MIT-BIH terms) splits are mandatory:
    intra-record splits leak rhythm structure across folds and inflate scores.
    """
    sets = {"train": set(train_records), "val": set(val_records), "test": set(test_records)}
    overlap = sets["train"] & sets["val"] | sets["train"] & sets["test"] | sets["val"] & sets["test"]
    if overlap:
        raise ValueError(f"Record IDs appear in multiple splits: {overlap}")

    train_idx, val_idx, test_idx = [], [], []
    for i, rid in enumerate(dataset.record_ids):
        if rid in sets["train"]:
            train_idx.append(i)
        elif rid in sets["val"]:
            val_idx.append(i)
        elif rid in sets["test"]:
            test_idx.append(i)
    return train_idx, val_idx, test_idx
