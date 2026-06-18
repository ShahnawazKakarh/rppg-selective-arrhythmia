"""MCD-rPPG dataset loader (post-inspection layout).

Loads the Hugging Face dataset `milai-oks-sakura/mcd_rppg` -- 3,600 synchronized
video recordings from 600 subjects with PPG, ECG, and 13 health biomarkers
across three camera views (frontal webcam, FullHD camcorder, mobile phone).
The original upload at `kyegorov/mcd_rppg` no longer resolves; the mirror is
the canonical location and is gated (requires accepting a contact-sharing
agreement on the dataset page).

On-disk layout (rooted at `data/raw/mcd_rppg/` after `snapshot_download`):

    db.csv                            -- master metadata table
    ecg/<id>_<step>.json              -- multi-lead clinical ECG (list of
                                         {title, values} channels + segmentsData)
    ppg/<id>_<step>.PW                -- binary contact-PPG (raw)
    ppg_sync/<id>_<camera>_<step>.txt -- PPG synchronized to video frames
                                         lines: "<ppg_value> <delta_seconds>"
    meta/<id>_<camera>_<step>.txt     -- per-frame timestamps
                                         lines: "<frame_idx> <iso_timestamp>"
    video/<id>_<camera>_<step>.avi    -- face video (~30 s, 30 fps)

Where:
    <id>     in {1020, 1024, ..., 9998}    -- 600 subjects
    <step>   in {before, after}            -- resting / post-exercise
    <camera> in {FullHDwebcam, USBVideo, IriunWebcam}

Reference:
    https://huggingface.co/datasets/milai-oks-sakura/mcd_rppg
    https://github.com/ksyegorov/mcd_rppg
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HF_REPO_ID = "milai-oks-sakura/mcd_rppg"


@dataclass
class MCDRecord:
    """One (subject, session, camera, view) record from MCD-rPPG.

    Attributes:
        patient_id: Subject identifier.
        step: "before" or "after" (resting vs. post-exercise).
        camera: "FullHDwebcam", "USBVideo", or "IriunWebcam".
        view: "front", "left", or "right".
        biomarkers: Dict of scalar biomarkers from db.csv (BP, SpO2, etc.).
        video_path: Path to the .avi face video.
        ecg_path: Path to the .json multi-lead clinical ECG.
        ppg_path: Path to the .PW binary contact-PPG file.
        ppg_sync_path: Path to the .txt synchronized PPG file.
        meta_path: Path to the .txt per-frame timestamp file.
    """

    patient_id: str
    step: str
    camera: str
    view: str
    biomarkers: dict[str, float]
    video_path: Path
    ecg_path: Path
    ppg_path: Path
    ppg_sync_path: Path
    meta_path: Path


def download(out_dir: str | Path = "data/raw/mcd_rppg", token: str | None = None) -> Path:
    """Snapshot-download MCD-rPPG from Hugging Face Hub.

    Requires accepting the gating agreement on
    https://huggingface.co/datasets/milai-oks-sakura/mcd_rppg first.
    """
    from huggingface_hub import snapshot_download

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(out_dir),
        token=token,
    )
    return Path(local)


def load_index(root: str | Path) -> pd.DataFrame:
    """Load `db.csv` as a DataFrame. One row per (subject, session, camera, view)."""
    root = Path(root)
    csv_path = root / "db.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run scripts/download_mcd_rppg.py first."
        )
    return pd.read_csv(csv_path)


def iter_samples(
    root: str | Path,
    patient_ids: list[str] | None = None,
    cameras: list[str] | None = None,
    require_files: bool = True,
) -> list[MCDRecord]:
    """Enumerate MCDRecord entries from db.csv, optionally filtered.

    Args:
        root: Directory containing the dataset (db.csv at root).
        patient_ids: Optional subject filter (strings or ints accepted).
        cameras: Optional camera filter, subset of {FullHDwebcam, USBVideo,
            IriunWebcam}.
        require_files: If True, only include records where ALL referenced
            files exist on disk (useful after a partial download).

    Returns:
        List of MCDRecord objects.
    """
    root = Path(root)
    df = load_index(root)

    if patient_ids is not None:
        ids = {str(p) for p in patient_ids}
        df = df[df["patient_id"].astype(str).isin(ids)]
    if cameras is not None:
        df = df[df["camera"].isin(cameras)]

    biomarker_cols = [
        c
        for c in df.columns
        if c
        not in {
            "patient_id",
            "step",
            "camera",
            "view",
            "ecg",
            "ppg",
            "video",
            "meta",
            "ppg_sync",
        }
    ]

    out: list[MCDRecord] = []
    for _, row in df.iterrows():
        biomarkers: dict[str, float] = {}
        for c in biomarker_cols:
            val = row[c]
            if pd.isna(val):
                biomarkers[c] = float("nan")
                continue
            try:
                biomarkers[c] = float(val)
            except (TypeError, ValueError):
                # Non-numeric biomarker (e.g. `sex` = 'F'/'M'); keep as-is.
                biomarkers[c] = val
        rec = MCDRecord(
            patient_id=str(row["patient_id"]),
            step=str(row["step"]),
            camera=str(row["camera"]),
            view=str(row["view"]),
            biomarkers=biomarkers,
            video_path=root / row["video"],
            ecg_path=root / row["ecg"],
            ppg_path=root / row["ppg"],
            ppg_sync_path=root / row["ppg_sync"],
            meta_path=root / row["meta"],
        )
        if require_files:
            paths = [rec.video_path, rec.ecg_path, rec.ppg_path, rec.ppg_sync_path, rec.meta_path]
            if not all(p.exists() for p in paths):
                continue
        out.append(rec)
    return out


def load_ppg_sync(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a synchronized PPG file.

    Empirical layout (confirmed via scripts/probe_ppg_sync.py):
      - The file has exactly one row per video frame.
      - Column 1 is the contact-PPG amplitude resampled to the video frame
        rate (~30 Hz), as an 8-bit unsigned integer (0..255).
      - Column 2 is a sync metadata field (timing offset between the underlying
        100 Hz PPG sample and the video frame). It is NOT inter-sample delta
        time; its values sum to far less than the recording duration.

    For HR estimation we treat column 1 as a uniformly-sampled time series at
    the video frame rate. The caller must supply the actual frame rate.

    Returns:
        values: (N,) array of PPG amplitudes, one per video frame.
        sync_offsets: (N,) array of the column-2 sync metadata (kept for
            optional fine alignment; most callers can ignore it).
    """
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected 2-column file at {path}; got shape {arr.shape}")
    return arr[:, 0], arr[:, 1]


def load_ecg_json(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Load a multi-lead ECG JSON file.

    Returns:
        lead_names: List of lead titles (e.g. ['V5', 'V1', ...]).
        signals: (n_leads, n_samples) float array.
    """
    import json

    with open(path) as f:
        d = json.load(f)
    # The top-level keys vary slightly across files; the channels live under
    # whichever key holds a list of {title, values} dicts.
    if isinstance(d, dict):
        # Find the list-of-channel-dicts value.
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "values" in v[0]:
                channels = v
                break
        else:
            raise ValueError(f"Could not locate channel list in {path}")
    elif isinstance(d, list) and d and isinstance(d[0], dict) and "values" in d[0]:
        channels = d
    else:
        raise ValueError(f"Unrecognized ECG JSON layout at {path}")

    lead_names = [str(ch.get("title", f"lead_{i}")) for i, ch in enumerate(channels)]
    # Pad or truncate to a common length (channels are usually equal length).
    lens = [len(ch["values"]) for ch in channels]
    n = min(lens) if min(lens) > 0 else max(lens)
    signals = np.stack([np.asarray(ch["values"][:n], dtype=np.float64) for ch in channels], axis=0)
    return lead_names, signals
