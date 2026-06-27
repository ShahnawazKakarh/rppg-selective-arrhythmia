"""Batch rPPG signal extraction from a directory of face videos.

Walks an input directory, runs MediaPipe ROI extraction + a chosen rPPG
algorithm (POS or CHROM) on each video, and writes per-video pulse waveform
CSVs + a single manifest CSV with metadata + signal-quality summaries.

Dataset-agnostic: works for MCD-rPPG, OBF, MAHNOB-HCI, or any directory of
videos. Pair the output manifest with a dataset-specific labels CSV
(record_id -> NSR/AF/Other, age, sex, etc.) downstream in the training
pipeline.

Output layout:
    <out_dir>/
        pulses/
            <video_basename>.csv        # columns: t_sec, pulse, fps
        manifest.csv                    # one row per video; columns include
                                        # filename, fps, n_frames, valid_frac,
                                        # method, mean_snr_db, peak_freq_hz,
                                        # peak_hr_bpm
        failures.csv                    # videos that could not be processed
                                        # (file not openable, no face found,
                                        # exception); columns include filename
                                        # and reason

Usage:
    python scripts/extract_rppg_batch.py \
        --videos-dir /path/to/videos \
        --out-dir runs/extract_obf \
        --method pos \
        --max-frames 900            # 30 s @ 30 fps; remove to use full clip

Designed to be called from a wrapper script per dataset (one for OBF, one
for MAHNOB, one for full-MCD-rPPG-600) that handles the dataset-specific
metadata join after extraction.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

# Repo path setup so this script runs without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rppg_sa.extractors.chrom import chrom  # noqa: E402
from rppg_sa.extractors.face_roi import extract_roi_signals, merge_rois  # noqa: E402
from rppg_sa.extractors.pos import pos  # noqa: E402
from rppg_sa.extractors.signal_quality import summarize_quality  # noqa: E402


def _peak_freq(pulse: np.ndarray, fs: float, lo_hz: float, hi_hz: float) -> float:
    """Dominant frequency (Hz) in the [lo_hz, hi_hz] band via Welch periodogram."""
    f, pxx = welch(pulse, fs=fs, nperseg=min(len(pulse), 512))
    band = (f >= lo_hz) & (f <= hi_hz)
    if not band.any():
        return 0.0
    return float(f[band][int(np.argmax(pxx[band]))])


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class ExtractionResult:
    filename: str
    fps: float
    n_frames: int
    valid_frac: float
    method: str
    mean_snr_db: float
    peak_freq_hz: float
    peak_hr_bpm: float
    pulse_csv: str
    elapsed_sec: float


def _extract_single(
    video_path: Path,
    out_dir: Path,
    method: str,
    max_frames: int | None,
    band_lo_hz: float,
    band_hi_hz: float,
) -> ExtractionResult:
    """Run the full single-video extraction. Raises on failure."""
    t0 = time.time()
    rois = extract_roi_signals(str(video_path), max_frames=max_frames)
    merged = merge_rois(rois)  # (T, 3) mean RGB across forehead + cheeks

    if method == "pos":
        pulse = pos(merged, rois.fps)
    elif method == "chrom":
        pulse = chrom(merged, rois.fps)
    else:
        raise ValueError(f"unknown method {method!r}")

    quality = summarize_quality(pulse, rois.fps)
    peak_hz = _peak_freq(pulse, rois.fps, band_lo_hz, band_hi_hz)
    n_frames = len(pulse)
    valid_frac = float(rois.valid_frames.mean())

    # Save pulse waveform.
    pulses_dir = out_dir / "pulses"
    pulses_dir.mkdir(parents=True, exist_ok=True)
    pulse_csv = pulses_dir / f"{video_path.stem}.csv"
    t_sec = np.arange(n_frames) / rois.fps
    pd.DataFrame({"t_sec": t_sec, "pulse": pulse}).to_csv(pulse_csv, index=False)

    return ExtractionResult(
        filename=video_path.name,
        fps=float(rois.fps),
        n_frames=int(n_frames),
        valid_frac=valid_frac,
        method=method,
        mean_snr_db=float(quality["snr_db"]),
        peak_freq_hz=float(peak_hz),
        peak_hr_bpm=float(peak_hz * 60.0),
        pulse_csv=str(pulse_csv.relative_to(out_dir)),
        elapsed_sec=time.time() - t0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos-dir", type=Path, required=True,
                        help="Directory containing face videos to process.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Destination directory for pulses/ and manifest.csv.")
    parser.add_argument("--method", choices=["pos", "chrom"], default="pos",
                        help="rPPG extraction algorithm (default: pos).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional cap on frames processed per video.")
    parser.add_argument("--band-lo-hz", type=float, default=0.7,
                        help="Low cutoff for the spectral peak search band (default: 0.7 Hz / 42 bpm).")
    parser.add_argument("--band-hi-hz", type=float, default=3.0,
                        help="High cutoff for the spectral peak search band (default: 3.0 Hz / 180 bpm).")
    parser.add_argument("--recursive", action="store_true",
                        help="Walk the videos directory recursively.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N videos (for smoke testing).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip videos already present in an existing manifest.csv.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate videos.
    if args.recursive:
        videos = sorted(
            p for ext in VIDEO_EXTENSIONS for p in args.videos_dir.rglob(f"*{ext}")
        )
    else:
        videos = sorted(
            p for ext in VIDEO_EXTENSIONS for p in args.videos_dir.glob(f"*{ext}")
        )
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        print(f"No video files found under {args.videos_dir}", file=sys.stderr)
        return 1

    # Resume support.
    manifest_path = args.out_dir / "manifest.csv"
    failures_path = args.out_dir / "failures.csv"
    done_filenames: set[str] = set()
    if args.resume and manifest_path.exists():
        try:
            existing = pd.read_csv(manifest_path)
            done_filenames = set(existing["filename"].astype(str).tolist())
            print(f"resume: skipping {len(done_filenames)} already-processed videos")
        except Exception as e:
            print(f"resume: could not read existing manifest ({e}); processing all")

    results: list[ExtractionResult] = []
    failures: list[dict] = []

    for i, video in enumerate(videos, start=1):
        if video.name in done_filenames:
            continue
        try:
            res = _extract_single(
                video, args.out_dir, args.method,
                args.max_frames, args.band_lo_hz, args.band_hi_hz,
            )
            results.append(res)
            print(f"[{i}/{len(videos)}] {video.name}  "
                  f"fps={res.fps:.1f}  n={res.n_frames}  "
                  f"valid={res.valid_frac:.2f}  hr={res.peak_hr_bpm:.1f}bpm  "
                  f"snr={res.mean_snr_db:.2f}dB  ({res.elapsed_sec:.1f}s)")
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            failures.append({"filename": video.name, "reason": reason})
            print(f"[{i}/{len(videos)}] {video.name}  FAILED  {reason}",
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # Persist manifest (append-safe under --resume).
    if results:
        new_df = pd.DataFrame([r.__dict__ for r in results])
        if args.resume and manifest_path.exists():
            old_df = pd.read_csv(manifest_path)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df.to_csv(manifest_path, index=False)
        print(f"\nWrote manifest: {manifest_path}  ({len(new_df)} rows total)")

    if failures:
        new_failures = pd.DataFrame(failures)
        if args.resume and failures_path.exists():
            old_failures = pd.read_csv(failures_path)
            new_failures = pd.concat([old_failures, new_failures], ignore_index=True)
        new_failures.to_csv(failures_path, index=False)
        print(f"Wrote failures: {failures_path}  ({len(new_failures)} rows total)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
