"""MIT-BIH Arrhythmia Database loader.

Loads ECG recordings from PhysioNet's MIT-BIH (record IDs 100..234) via the
`wfdb` package, segments by rhythm-annotation regions, and exposes 30-second
windows labelled as NSR / AF / Other for the synthesis-based fallback path.

Reference:
    Moody & Mark (2001). "The impact of the MIT-BIH Arrhythmia Database."
    IEEE Engineering in Medicine and Biology, 20(3):45-50.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# MIT-BIH rhythm annotation codes -> our 3-class scheme.
# (See https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm)
RHYTHM_TO_CLASS = {
    "(N": "NSR",        # Normal sinus rhythm
    "(AFIB": "AF",      # Atrial fibrillation
    "(AFL": "Other",    # Atrial flutter
    "(SVTA": "Other",   # Supraventricular tachyarrhythmia
    "(VT": "Other",     # Ventricular tachycardia
    "(B": "Other",      # Ventricular bigeminy
    "(T": "Other",      # Ventricular trigeminy
    "(IVR": "Other",    # Idioventricular rhythm
    "(VFL": "Other",    # Ventricular flutter
}
CLASS_TO_INDEX = {"NSR": 0, "AF": 1, "Other": 2}


def load_record(record_path: str | Path) -> tuple[np.ndarray, float, list[tuple[int, str]]]:
    """Load an MIT-BIH record and its rhythm annotations.

    Args:
        record_path: Path stem WITHOUT extension (e.g. `data/raw/mitbih/100`).

    Returns:
        signal: (T,) array, first channel (MLII lead by convention).
        fs: Sampling rate in Hz (360 for MIT-BIH).
        rhythm_segments: List of (sample_index, rhythm_code) tuples ordered by
            sample_index; defines the rhythm label active from each index
            forward until the next entry.
    """
    import wfdb

    record_path = str(record_path)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, extension="atr")

    signal = record.p_signal[:, 0]
    fs = float(record.fs)

    rhythm_segments: list[tuple[int, str]] = []
    for sample, aux in zip(ann.sample, ann.aux_note):
        if aux and aux.startswith("("):
            # `aux_note` includes a trailing '\x00' on some records; strip it.
            code = aux.rstrip("\x00").strip()
            rhythm_segments.append((int(sample), code))

    return signal, fs, rhythm_segments


def segment_record(
    signal: np.ndarray,
    fs: float,
    rhythm_segments: list[tuple[int, str]],
    window_seconds: float = 30.0,
    step_seconds: float = 30.0,
) -> list[dict]:
    """Slice a record into fixed-length windows with rhythm labels.

    Args:
        signal: 1-D ECG signal.
        fs: Sampling rate in Hz.
        rhythm_segments: As returned by `load_record`.
        window_seconds: Window length in seconds.
        step_seconds: Hop between consecutive windows.

    Returns:
        List of dicts: {start, end, label_idx, label_str, signal}.
    """
    win = int(round(window_seconds * fs))
    step = int(round(step_seconds * fs))
    out: list[dict] = []

    # Build a function that maps sample-index -> active rhythm code.
    if not rhythm_segments:
        return out
    starts = np.array([s for s, _ in rhythm_segments], dtype=np.int64)
    codes = [c for _, c in rhythm_segments]

    def rhythm_at(idx: int) -> str:
        pos = int(np.searchsorted(starts, idx, side="right")) - 1
        return codes[max(pos, 0)]

    for s in range(0, len(signal) - win + 1, step):
        e = s + win
        code = rhythm_at(s)
        label_str = RHYTHM_TO_CLASS.get(code, "Other")
        out.append(
            {
                "start": s,
                "end": e,
                "label_str": label_str,
                "label_idx": CLASS_TO_INDEX[label_str],
                "signal": signal[s:e],
            }
        )
    return out


def load_dataset(
    root: str | Path,
    record_ids: list[str] | None = None,
    window_seconds: float = 30.0,
    step_seconds: float = 30.0,
) -> list[dict]:
    """Load all (or selected) MIT-BIH records and return windowed segments.

    Args:
        root: Directory containing MIT-BIH records (e.g. `data/raw/mitbih`).
        record_ids: Optional subset of record IDs (strings like "100", "101").
            If None, auto-discover by scanning `*.hea` files.
        window_seconds: Window length passed through to `segment_record`.
        step_seconds: Window hop passed through to `segment_record`.
    """
    root = Path(root)
    if record_ids is None:
        record_ids = sorted({p.stem for p in root.glob("*.hea")})

    all_segments: list[dict] = []
    for rid in record_ids:
        sig, fs, rhy = load_record(root / rid)
        segs = segment_record(
            sig,
            fs,
            rhy,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )
        for s in segs:
            s["record_id"] = rid
            s["fs"] = fs
        all_segments.extend(segs)
    return all_segments
