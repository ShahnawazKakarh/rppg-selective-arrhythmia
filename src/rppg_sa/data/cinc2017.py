"""PhysioNet/CinC 2017 AF Classification Challenge loader.

Layout (after running scripts/download_cinc2017.py):

    data/raw/cinc2017/
      training2017/
        A00001.mat    # ECG signal (key 'val', shape (1, T) int16)
        A00001.hea    # header
        ...
        REFERENCE.csv # record_id,label  with label ∈ {N, A, O, ~}

Labels map to our 3-class task:
    N -> NSR
    A -> AF
    O -> Other
    ~ -> dropped (too noisy to classify)

Sampling rate is 300 Hz across all records, signal length varies (9-60 s).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


CINC_LABEL_TO_CLASS: dict[str, str] = {
    "N": "NSR",
    "A": "AF",
    "O": "Other",
    "~": "Noisy",   # dropped before training
}
CINC_FS: float = 300.0


@dataclass
class CinCRecord:
    record_id: str
    label_raw: str       # 'N' / 'A' / 'O' / '~'
    label_class: str     # 'NSR' / 'AF' / 'Other' / 'Noisy'
    mat_path: Path


def _read_reference(reference_csv: Path) -> list[tuple[str, str]]:
    """Returns list of (record_id, label_raw)."""
    rows: list[tuple[str, str]] = []
    with reference_csv.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rid, label = line.split(",", 1)
            rows.append((rid.strip(), label.strip()))
    return rows


def iter_records(
    root: Path, include_noisy: bool = False
) -> Iterator[CinCRecord]:
    """Iterate CinC records under <root>/training2017/.

    Drops noisy ('~') records unless include_noisy=True.
    """
    training_dir = Path(root) / "training2017"
    reference_csv = training_dir / "REFERENCE.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(
            f"REFERENCE.csv not found at {reference_csv}. "
            "Run scripts/download_cinc2017.py first."
        )
    for rid, label_raw in _read_reference(reference_csv):
        label_class = CINC_LABEL_TO_CLASS.get(label_raw, "Other")
        if label_class == "Noisy" and not include_noisy:
            continue
        mat_path = training_dir / f"{rid}.mat"
        if not mat_path.exists():
            continue
        yield CinCRecord(
            record_id=rid,
            label_raw=label_raw,
            label_class=label_class,
            mat_path=mat_path,
        )


def load_ecg(record: CinCRecord) -> tuple[np.ndarray, float]:
    """Load a single CinC record's ECG signal as (1-D float32, fs)."""
    from scipy.io import loadmat

    mat = loadmat(record.mat_path)
    if "val" not in mat:
        raise ValueError(f"{record.mat_path} has no 'val' key")
    ecg = np.asarray(mat["val"], dtype=np.float32).squeeze()
    if ecg.ndim != 1:
        raise ValueError(f"{record.mat_path} ECG not 1-D after squeeze: shape {ecg.shape}")
    return ecg, CINC_FS


def label_distribution(root: Path) -> dict[str, int]:
    """Quick count by class label (Noisy included)."""
    counts: dict[str, int] = {}
    for rec in iter_records(root, include_noisy=True):
        counts[rec.label_class] = counts.get(rec.label_class, 0) + 1
    return counts
