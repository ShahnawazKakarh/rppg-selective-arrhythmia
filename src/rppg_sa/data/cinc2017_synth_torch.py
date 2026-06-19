"""PyTorch Dataset: synthesized rPPG from CinC 2017 AF Challenge ECGs.

Same synthesis pipeline as synth_rppg_torch (R-peak detection + beat template
+ downsample + noise), but the input is CinC 2017 (8,528 records, ~10x AF
labels vs MIT-BIH). Records are short (9-60 s), so we typically get 0-2
windows per record at the 30 s default. Reducing window_seconds to 10 s
yields more samples per AF record, which is the right move for this dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rppg_sa.data.cinc2017 import CINC_FS, CinCRecord, iter_records, load_ecg
from rppg_sa.data.mitbih import CLASS_TO_INDEX
from rppg_sa.data.synth_rppg import synth_ppg_from_ecg


class CinCSynthRPPGSegmentDataset(Dataset):
    """Synth-rPPG segments from CinC 2017. Each record carries one label."""

    LABEL_NAMES = ["NSR", "AF", "Other"]

    def __init__(
        self,
        root: str | Path,
        target_fs: float = 30.0,
        window_seconds: float = 10.0,
        step_seconds: float | None = None,
        cache_dir: str | Path | None = None,
        synth_seed: int = 42,
        noise_sigma: float = 0.05,
        motion_burst_prob: float = 0.0,
        lighting_flicker_amp: float = 0.0,
        record_ids: list[str] | None = None,
        max_records: int | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.target_fs = float(target_fs)
        self.window_seconds = float(window_seconds)
        self.step_seconds = float(step_seconds) if step_seconds is not None else self.window_seconds
        self.win_samples = int(round(self.target_fs * self.window_seconds))
        self.step_samples = int(round(self.target_fs * self.step_seconds))
        if cache_dir is None:
            cache_dir = self.root.parent.parent / "processed" / "synth_rppg_cinc"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.synth_seed = int(synth_seed)
        self.noise_sigma = float(noise_sigma)
        self.motion_burst_prob = float(motion_burst_prob)
        self.lighting_flicker_amp = float(lighting_flicker_amp)

        wanted_ids = set(record_ids) if record_ids is not None else None
        records: list[CinCRecord] = []
        for rec in iter_records(self.root, include_noisy=False):
            if wanted_ids is not None and rec.record_id not in wanted_ids:
                continue
            records.append(rec)
            if max_records is not None and len(records) >= max_records:
                break

        self.signals: list[np.ndarray] = []
        self.labels: list[int] = []
        self.record_ids: list[str] = []

        n_cached = 0
        n_synth = 0
        n_skipped = 0
        for rec in records:
            try:
                pulse = self._load_or_synth(rec)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                print(f"  skip {rec.record_id}: {exc}")
                n_skipped += 1
                continue

            if pulse is None or len(pulse) < self.win_samples:
                n_skipped += 1
                continue

            label_idx = CLASS_TO_INDEX[rec.label_class]
            for s in range(0, len(pulse) - self.win_samples + 1, self.step_samples):
                self.signals.append(pulse[s : s + self.win_samples].astype(np.float32))
                self.labels.append(label_idx)
                self.record_ids.append(rec.record_id)

            if (self.cache_dir / self._cache_name(rec.record_id)).exists():
                n_cached += 1
            else:
                n_synth += 1

        print(
            f"CinCSynthRPPGSegmentDataset: {len(self.signals)} segments from "
            f"{len(set(self.record_ids))} records "
            f"(cache hit {n_cached}, synthesized {n_synth}, skipped {n_skipped})"
        )

    def _cache_name(self, rid: str) -> str:
        return f"{rid}_fs{int(self.target_fs)}_seed{self.synth_seed}_pos.npy"

    def _load_or_synth(self, rec: CinCRecord) -> np.ndarray:
        cache_path = self.cache_dir / self._cache_name(rec.record_id)
        if cache_path.exists():
            return np.load(cache_path)
        ecg, fs_ecg = load_ecg(rec)
        rng = np.random.default_rng(self.synth_seed + hash(rec.record_id) % (2 ** 31))
        pulse = synth_ppg_from_ecg(
            ecg,
            fs_ecg=fs_ecg,
            fs_out=self.target_fs,
            noise_sigma=self.noise_sigma,
            motion_burst_prob=self.motion_burst_prob,
            lighting_flicker_amp=self.lighting_flicker_amp,
            rng=rng,
        )
        np.save(cache_path, pulse)
        return pulse

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "signal": torch.from_numpy(self.signals[idx]).unsqueeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "record_id": self.record_ids[idx],
        }

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.LABEL_NAMES}
        for y in self.labels:
            counts[self.LABEL_NAMES[y]] += 1
        return counts


def auto_split_records(
    root: str | Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministic 70/15/15 record-level split, stratified by class label.

    Each record contributes either to train, val, or test (no record-level leakage).
    Stratification ensures all three classes are represented in every split.
    """
    import random

    by_label: dict[str, list[str]] = {"NSR": [], "AF": [], "Other": []}
    for rec in iter_records(Path(root), include_noisy=False):
        by_label[rec.label_class].append(rec.record_id)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for cls, ids in by_label.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test += ids[:n_test]
        val += ids[n_test : n_test + n_val]
        train += ids[n_test + n_val :]
    return sorted(train), sorted(val), sorted(test)


def subject_disjoint_split(
    dataset: CinCSynthRPPGSegmentDataset,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> tuple[list[int], list[int], list[int]]:
    sets = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    overlap = sets["train"] & sets["val"] | sets["train"] & sets["test"] | sets["val"] & sets["test"]
    if overlap:
        raise ValueError(f"Records appear in multiple splits: {overlap}")
    train_idx, val_idx, test_idx = [], [], []
    for i, rid in enumerate(dataset.record_ids):
        if rid in sets["train"]:
            train_idx.append(i)
        elif rid in sets["val"]:
            val_idx.append(i)
        elif rid in sets["test"]:
            test_idx.append(i)
    return train_idx, val_idx, test_idx
