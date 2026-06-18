"""PyTorch Dataset: synthesized rPPG from MIT-BIH ECG.

Mirrors MITBIHSegmentDataset's API so train_classifier.py and eval_selective.py
work without changes beyond a `source: synth_rppg` branch.

Caches synthesized waveforms per (record_id, fs_out, seed) so first epoch is
slow (R-peak detection + synthesis), subsequent epochs are instant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rppg_sa.data.mitbih import (
    CLASS_TO_INDEX,
    RHYTHM_TO_CLASS,
    load_record,
    segment_record,
)
from rppg_sa.data.synth_rppg import synth_ppg_from_ecg


class SynthRPPGSegmentDataset(Dataset):
    """rPPG-style pulse synthesized from MIT-BIH ECG, with NSR/AF/Other labels.

    Args:
        root: MIT-BIH root (contains *.hea files).
        record_ids: Subset of record IDs (e.g. ["100", "101"]).
        target_fs: Output (rPPG) sampling rate. Default 30 Hz (camera fps).
        window_seconds: Segment length. 30 s is standard.
        step_seconds: Window hop.
        cache_dir: Where synthesized .npy files are written.
        synth_seed: Seed for the deterministic synthesizer.
        noise_sigma: Gaussian noise std (see synth_rppg).
        motion_burst_prob: Probability of a motion burst per second.
        lighting_flicker_amp: Amplitude of 4-Hz lighting flicker.
    """

    LABEL_NAMES = ["NSR", "AF", "Other"]

    def __init__(
        self,
        root: str | Path,
        record_ids: list[str],
        target_fs: float = 30.0,
        window_seconds: float = 30.0,
        step_seconds: float = 30.0,
        cache_dir: str | Path | None = None,
        synth_seed: int = 42,
        noise_sigma: float = 0.05,
        motion_burst_prob: float = 0.0,
        lighting_flicker_amp: float = 0.0,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.target_fs = float(target_fs)
        self.window_seconds = float(window_seconds)
        self.step_seconds = float(step_seconds)
        self.win_samples = int(round(self.target_fs * self.window_seconds))
        self.step_samples = int(round(self.target_fs * self.step_seconds))
        if cache_dir is None:
            cache_dir = self.root.parent.parent / "processed" / "synth_rppg"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.synth_seed = int(synth_seed)
        self.noise_sigma = float(noise_sigma)
        self.motion_burst_prob = float(motion_burst_prob)
        self.lighting_flicker_amp = float(lighting_flicker_amp)

        self.signals: list[np.ndarray] = []
        self.labels: list[int] = []
        self.record_ids: list[str] = []

        n_extracted = 0
        n_cached = 0
        for rid in record_ids:
            try:
                pulse, fs_ecg, rhythm_segments = self._load_or_synth(rid)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"  skip {rid}: {exc}")
                continue

            if (self.cache_dir / f"{rid}_pos.npy").exists():
                n_cached += 1
            else:
                n_extracted += 1

            # Map ECG-rate rhythm boundaries to target_fs sample indices.
            scale = self.target_fs / fs_ecg
            scaled = [(int(round(s * scale)), c) for s, c in rhythm_segments]

            # Window the synthesized pulse with rhythm labels from MIT-BIH.
            if not scaled:
                continue
            starts = np.array([s for s, _ in scaled], dtype=np.int64)
            codes = [c for _, c in scaled]

            def rhythm_at(idx: int) -> str:
                pos = int(np.searchsorted(starts, idx, side="right")) - 1
                return codes[max(pos, 0)]

            for s in range(0, len(pulse) - self.win_samples + 1, self.step_samples):
                code = rhythm_at(s)
                label_str = RHYTHM_TO_CLASS.get(code, "Other")
                self.signals.append(pulse[s : s + self.win_samples].astype(np.float32))
                self.labels.append(CLASS_TO_INDEX[label_str])
                self.record_ids.append(rid)

        print(
            f"SynthRPPGSegmentDataset: {len(self.signals)} segments from "
            f"{len(set(self.record_ids))} records "
            f"(cache hit {n_cached}, synthesized {n_extracted})"
        )

    def _cache_name(self, rid: str) -> str:
        return f"{rid}_fs{int(self.target_fs)}_seed{self.synth_seed}_pos.npy"

    def _load_or_synth(self, rid: str) -> tuple[np.ndarray, float, list[tuple[int, str]]]:
        ecg, fs_ecg, rhythm_segments = load_record(self.root / rid)
        cache_path = self.cache_dir / self._cache_name(rid)
        if cache_path.exists():
            pulse = np.load(cache_path)
            return pulse, fs_ecg, rhythm_segments

        rng = np.random.default_rng(self.synth_seed + int(rid))
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
        return pulse, fs_ecg, rhythm_segments

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


def subject_disjoint_split(
    dataset: SynthRPPGSegmentDataset,
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
