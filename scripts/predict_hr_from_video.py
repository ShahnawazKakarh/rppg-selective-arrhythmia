"""Predict heart rate from a face video using a classical rPPG extractor.

Pipeline:
    face video -> MediaPipe FaceMesh ROIs -> mean RGB per ROI -> merge ->
    CHROM or POS extractor -> spectral peak -> bpm

Usage:
    python scripts/predict_hr_from_video.py --video clip.mp4 --method chrom
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import welch

from rppg_sa.extractors.chrom import chrom
from rppg_sa.extractors.face_roi import extract_roi_signals, merge_rois
from rppg_sa.extractors.pos import pos
from rppg_sa.extractors.signal_quality import summarize_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict heart rate from a face video.")
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--method",
        choices=["chrom", "pos"],
        default="chrom",
        help="rPPG extraction method.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames processed.",
    )
    parser.add_argument(
        "--save-pulse",
        type=Path,
        default=None,
        help="Optional .npy path to save the extracted pulse waveform.",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    print(f"Extracting ROIs from {args.video} ...")
    rois = extract_roi_signals(str(args.video), max_frames=args.max_frames)
    n_valid = int(rois.valid_frames.sum())
    print(f"  fps={rois.fps:.2f}, valid face frames: {n_valid}/{len(rois.valid_frames)}")
    if n_valid < 0.5 * len(rois.valid_frames):
        print("  WARNING: less than half the frames had a detectable face.")

    rgb = merge_rois(rois)
    if args.method == "chrom":
        pulse = chrom(rgb, fs=rois.fps)
    else:
        pulse = pos(rgb, fs=rois.fps)

    # Spectral peak -> heart rate.
    f, pxx = welch(pulse, fs=rois.fps, nperseg=min(len(pulse), 512))
    band = (f >= 0.7) & (f <= 4.0)
    f_band = f[band]
    p_band = pxx[band]
    hr_bpm = float(f_band[int(np.argmax(p_band))] * 60.0)

    quality = summarize_quality(pulse, fs=rois.fps)

    print(f"\nMethod: {args.method.upper()}")
    print(f"Heart rate: {hr_bpm:.1f} bpm")
    print(f"Signal quality:")
    print(f"  spectral SNR : {quality['snr_db']:.2f} dB")
    print(f"  template SQI : {quality['template_sqi']:.3f}")

    if args.save_pulse:
        args.save_pulse.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_pulse, pulse)
        print(f"Saved pulse waveform to {args.save_pulse}")


if __name__ == "__main__":
    main()
