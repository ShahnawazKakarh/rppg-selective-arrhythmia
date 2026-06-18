"""Validate rPPG extractors against MCD-rPPG ground truth.

For one or more (subject, session, camera) records, run CHROM and POS on the
face video and compare the recovered pulse against the ground-truth PPG
(`ppg_sync/*.txt`). Reports heart-rate error in bpm and signal-quality stats.

This is the first end-to-end rPPG validation in the repo and produces the
first real heart-rate numbers for the README's Findings section.

Usage:
    python scripts/validate_rppg_on_mcd.py \\
        --root data/raw/mcd_rppg \\
        --patient 1020 \\
        --camera FullHDwebcam
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch

from rppg_sa.data.mcd_rppg import iter_samples, load_ppg_sync
from rppg_sa.extractors.chrom import chrom
from rppg_sa.extractors.face_roi import extract_roi_signals, merge_rois
from rppg_sa.extractors.hr import windowed_hr_bpm
from rppg_sa.extractors.pos import pos
from rppg_sa.extractors.signal_quality import summarize_quality


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 4.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def _resample_to_uniform(values: np.ndarray, timestamps: np.ndarray, fs_out: float = 30.0) -> tuple[np.ndarray, float]:
    """Interpolate irregularly-sampled (values, timestamps) onto a uniform grid."""
    t_start, t_end = float(timestamps[0]), float(timestamps[-1])
    n = max(int((t_end - t_start) * fs_out), 16)
    t_uniform = np.linspace(t_start, t_end, n)
    interp = interp1d(timestamps, values, kind="linear", fill_value="extrapolate")
    return interp(t_uniform), fs_out


def evaluate_record(
    record,
    max_frames: int | None = None,
) -> dict:
    """Run CHROM and POS on a single MCDRecord, compare to ground-truth HR.

    Two ground-truth references:
      - `biomarker_hr_bpm`: the clinically recorded pulse from db.csv. Most
        reliable single number; primary scoring target.
      - `ppg_sync_hr_bpm`: spectral peak from the per-frame synchronized PPG
        file. Useful as a sanity check but its column format is still under
        investigation (see scripts/probe_ppg_sync.py).
    """
    # ---------- Primary ground truth: biomarker pulse from db.csv ----------
    biomarker_hr = record.biomarkers.get("pulse")
    if biomarker_hr is None or (isinstance(biomarker_hr, float) and np.isnan(biomarker_hr)):
        biomarker_hr = float("nan")
    else:
        biomarker_hr = float(biomarker_hr)

    # ---------- rPPG extraction ----------
    rois = extract_roi_signals(str(record.video_path), max_frames=max_frames)
    valid_frac = float(rois.valid_frames.sum()) / max(len(rois.valid_frames), 1)
    n_frames = len(rois.valid_frames)
    duration_s = n_frames / rois.fps if rois.fps else 0.0
    print(
        f"     video: {n_frames} frames @ {rois.fps:.2f} fps -> {duration_s:.1f}s, "
        f"face-detect {valid_frac*100:.1f}%"
    )
    rgb = merge_rois(rois)

    # ---------- Secondary ground truth: synchronized PPG file ----------
    # ppg_sync has one row per video frame; we treat column 1 as the PPG signal
    # sampled at the video frame rate (~30 Hz).
    gt_values, _ = load_ppg_sync(record.ppg_sync_path)
    # Trim to whichever is shorter, in case of one-frame drift.
    common_n = min(len(gt_values), n_frames)
    gt_values = gt_values[:common_n]
    gt_fs = rois.fps
    gt_filtered = _bandpass(gt_values - gt_values.mean(), fs=gt_fs)
    gt_windowed = windowed_hr_bpm(gt_filtered, fs=gt_fs)
    ppg_sync_hr = gt_windowed["hr_bpm"]
    gt_quality = summarize_quality(gt_filtered, fs=gt_fs)

    pulse_chrom = chrom(rgb, fs=rois.fps)
    pulse_pos = pos(rgb, fs=rois.fps)
    chrom_windowed = windowed_hr_bpm(pulse_chrom, fs=rois.fps)
    pos_windowed = windowed_hr_bpm(pulse_pos, fs=rois.fps)
    hr_chrom = chrom_windowed["hr_bpm"]
    hr_pos = pos_windowed["hr_bpm"]
    q_chrom = summarize_quality(pulse_chrom, fs=rois.fps)
    q_pos = summarize_quality(pulse_pos, fs=rois.fps)

    def err(est: float, gt: float) -> float:
        return float("nan") if np.isnan(gt) else float(est - gt)

    def abs_err(est: float, gt: float) -> float:
        return float("nan") if np.isnan(gt) else float(abs(est - gt))

    return {
        "patient_id": record.patient_id,
        "step": record.step,
        "camera": record.camera,
        "view": record.view,
        "video_fps": rois.fps,
        "video_n_frames": n_frames,
        "video_duration_s": duration_s,
        "face_detection_rate": valid_frac,
        "biomarker_hr_bpm": biomarker_hr,
        "ppg_sync": {
            "hr_bpm": ppg_sync_hr,
            "hr_std_bpm": gt_windowed["hr_std_bpm"],
            "n_windows": gt_windowed["n_windows"],
            "snr_db": gt_quality["snr_db"],
            "template_sqi": gt_quality["template_sqi"],
            "n_samples": int(common_n),
            "sampled_at_video_fps": float(gt_fs),
        },
        "chrom": {
            "hr_bpm": hr_chrom,
            "hr_std_bpm": chrom_windowed["hr_std_bpm"],
            "n_windows": chrom_windowed["n_windows"],
            "hr_error_vs_ppg_sync_bpm": float(hr_chrom - ppg_sync_hr),
            "abs_hr_error_vs_ppg_sync_bpm": float(abs(hr_chrom - ppg_sync_hr)),
            "hr_error_vs_biomarker_bpm": err(hr_chrom, biomarker_hr),
            "abs_hr_error_vs_biomarker_bpm": abs_err(hr_chrom, biomarker_hr),
            "snr_db": q_chrom["snr_db"],
            "template_sqi": q_chrom["template_sqi"],
        },
        "pos": {
            "hr_bpm": hr_pos,
            "hr_std_bpm": pos_windowed["hr_std_bpm"],
            "n_windows": pos_windowed["n_windows"],
            "hr_error_vs_ppg_sync_bpm": float(hr_pos - ppg_sync_hr),
            "abs_hr_error_vs_ppg_sync_bpm": float(abs(hr_pos - ppg_sync_hr)),
            "hr_error_vs_biomarker_bpm": err(hr_pos, biomarker_hr),
            "abs_hr_error_vs_biomarker_bpm": abs_err(hr_pos, biomarker_hr),
            "snr_db": q_pos["snr_db"],
            "template_sqi": q_pos["template_sqi"],
        },
        "biomarkers": {
            "age": record.biomarkers.get("age"),
            "sex": record.biomarkers.get("sex"),
            "bmi": record.biomarkers.get("bmi"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CHROM/POS against MCD-rPPG ground truth.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/mcd_rppg"))
    parser.add_argument("--patient", nargs="+", default=["1020"], help="One or more patient IDs.")
    parser.add_argument(
        "--camera",
        nargs="+",
        default=["FullHDwebcam"],
        choices=["FullHDwebcam", "USBVideo", "IriunWebcam"],
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/baselines/mcd_rppg_v0/results.json"),
    )
    args = parser.parse_args()

    records = iter_samples(args.root, patient_ids=args.patient, cameras=args.camera)
    if not records:
        raise SystemExit(
            f"No records matched patient={args.patient}, camera={args.camera}. "
            "Use scripts/download_mcd_rppg.py with the right --allow-patterns to fetch them."
        )

    print(f"Evaluating {len(records)} record(s) ...")
    results = []
    for r in records:
        print(f"  -> {r.patient_id} | {r.step} | {r.camera} | {r.view}")
        res = evaluate_record(r, max_frames=args.max_frames)
        bio = res["biomarker_hr_bpm"]
        ppg = res["ppg_sync"]["hr_bpm"]
        ce = res["chrom"]["hr_error_vs_ppg_sync_bpm"]
        pe = res["pos"]["hr_error_vs_ppg_sync_bpm"]
        print(
            f"     PPG-sync GT {ppg:6.2f} bpm | Biomarker {bio:6.2f} (reference) | "
            f"CHROM {res['chrom']['hr_bpm']:6.2f} (\u0394 {ce:+6.2f}) | "
            f"POS {res['pos']['hr_bpm']:6.2f} (\u0394 {pe:+6.2f})"
        )
        results.append(res)

    # Aggregate against PPG-sync GT (primary, synchronized to video time window).
    chrom_mae = float(np.mean([r["chrom"]["abs_hr_error_vs_ppg_sync_bpm"] for r in results]))
    pos_mae = float(np.mean([r["pos"]["abs_hr_error_vs_ppg_sync_bpm"] for r in results]))
    # Also aggregate against biomarker for reference (subject to cuff-vs-video timing mismatch).
    chrom_errs_bio = [r["chrom"]["abs_hr_error_vs_biomarker_bpm"] for r in results]
    pos_errs_bio = [r["pos"]["abs_hr_error_vs_biomarker_bpm"] for r in results]
    chrom_errs_bio = [e for e in chrom_errs_bio if not (isinstance(e, float) and np.isnan(e))]
    pos_errs_bio = [e for e in pos_errs_bio if not (isinstance(e, float) and np.isnan(e))]
    chrom_mae_bio = float(np.mean(chrom_errs_bio)) if chrom_errs_bio else float("nan")
    pos_mae_bio = float(np.mean(pos_errs_bio)) if pos_errs_bio else float("nan")

    summary = {
        "n_records": len(results),
        "primary_ground_truth": "ppg_sync",
        "chrom_mae_vs_ppg_sync_bpm": chrom_mae,
        "pos_mae_vs_ppg_sync_bpm": pos_mae,
        "chrom_mae_vs_biomarker_bpm": chrom_mae_bio,
        "pos_mae_vs_biomarker_bpm": pos_mae_bio,
        "per_record": results,
    }
    print(
        f"\nSummary vs PPG-sync (primary GT): "
        f"CHROM MAE {chrom_mae:.2f} bpm | POS MAE {pos_mae:.2f} bpm  (n={len(results)})"
    )
    print(
        f"Summary vs biomarker pulse (reference): "
        f"CHROM MAE {chrom_mae_bio:.2f} bpm | POS MAE {pos_mae_bio:.2f} bpm"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
