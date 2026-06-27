"""Validate POS/CHROM rPPG extractor against MCD-rPPG reference PPG waveforms.

Uses windowed_hr_bpm (10s windows, 50% overlap, median) on BOTH the extracted
POS pulse and the ppg_sync reference, matching the canonical methodology in
src/rppg_sa/extractors/hr.py. Single-clip Welch peak is unreliable here:
non-stationary HR during post-exercise recovery and sub-harmonic lock-on at
high HR both push large errors when only one peak is taken over the full clip.

Inputs:
  - runs/extract_mcd_full/manifest.csv (filename + fps per video, from
    scripts/extract_rppg_batch.py)
  - runs/extract_mcd_full/pulses/<stem>.csv (POS pulse waveforms per video)
  - data/raw/mcd_rppg/db.csv (subject metadata)
  - data/raw/mcd_rppg/ppg_sync/<stem>.txt (per-frame reference PPG; col 0 is
    PPG amplitude, sampled at video fps per src/rppg_sa/data/mcd_rppg.py)

Outputs:
  - <out_dir>/per_row.csv (per-video extracted vs reference HR + per-window
    arrays)
  - <out_dir>/summary.json (MAE, RMSE, median abs err, Pearson r)
  - <out_dir>/per_step.csv (stratification: before vs after exercise)
  - <out_dir>/per_snr_tertile.csv (stratification: SNR tertile from extraction
    manifest, useful for LW-CCSD-style mechanism analysis)

Usage:
    python scripts/validate_rppg_on_mcd_full.py \\
        --manifest runs/extract_mcd_full/manifest.csv \\
        --out-dir runs/validate_mcd_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rppg_sa.extractors.hr import windowed_hr_bpm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to manifest.csv from extract_rppg_batch.py.")
    parser.add_argument("--db-csv", type=Path,
                        default=Path("data/raw/mcd_rppg/db.csv"),
                        help="Path to MCD-rPPG db.csv.")
    parser.add_argument("--ppg-sync-dir", type=Path,
                        default=Path("data/raw/mcd_rppg/ppg_sync"),
                        help="Directory of per-video ppg_sync .txt files.")
    parser.add_argument("--pulses-dir", type=Path, default=None,
                        help="Directory of extracted pulse CSVs (defaults to "
                             "<manifest_dir>/pulses).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Destination directory for validation outputs.")
    parser.add_argument("--window-seconds", type=float, default=10.0,
                        help="HR window length (default: 10 s).")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Fractional overlap between HR windows (default: 0.5).")
    parser.add_argument("--band-lo-hz", type=float, default=0.7,
                        help="Low cutoff for the HR band (default: 0.7 Hz).")
    parser.add_argument("--band-hi-hz", type=float, default=3.0,
                        help="High cutoff for the HR band (default: 3.0 Hz).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pulses_dir = args.pulses_dir or args.manifest.parent / "pulses"

    manifest = pd.read_csv(args.manifest)
    db = pd.read_csv(args.db_csv)
    print(f"Loaded manifest: {len(manifest)} rows")
    print(f"Loaded db.csv: {len(db)} rows ({db['patient_id'].nunique()} unique patients)")

    def parse_filename(fn: str) -> tuple[int, str, str]:
        stem = Path(fn).stem
        parts = stem.split("_")
        return int(parts[0]), parts[1], parts[2]

    parsed = manifest["filename"].apply(parse_filename).tolist()
    manifest["patient_id"] = [p[0] for p in parsed]
    manifest["camera"] = [p[1] for p in parsed]
    manifest["step"] = [p[2] for p in parsed]

    joined = manifest.merge(
        db[["patient_id", "camera", "step", "age", "sex", "pulse"]],
        on=["patient_id", "camera", "step"],
        how="left",
    )
    print(f"Joined to db.csv: {joined['age'].notna().sum()}/{len(joined)} matched")

    band = (args.band_lo_hz, args.band_hi_hz)

    extracted_hr: list[float] = []
    reference_hr: list[float] = []
    extracted_hr_per_window: list[list[float]] = []
    reference_hr_per_window: list[list[float]] = []

    for _, row in joined.iterrows():
        stem = Path(row["filename"]).stem
        fs = float(row["fps"])

        # Extracted HR — recompute from the saved pulse waveform with
        # windowed median (the manifest's peak_hr_bpm is a single-clip Welch
        # peak; we replace it).
        pulse_csv = pulses_dir / f"{stem}.csv"
        if pulse_csv.exists():
            pulse = pd.read_csv(pulse_csv)["pulse"].to_numpy()
            ext_res = windowed_hr_bpm(
                pulse, fs,
                window_seconds=args.window_seconds,
                overlap=args.overlap,
                hr_band=band,
            )
            extracted_hr.append(ext_res["hr_bpm"])
            extracted_hr_per_window.append(ext_res["hr_bpm_per_window"])
        else:
            extracted_hr.append(np.nan)
            extracted_hr_per_window.append([])

        # Reference HR from ppg_sync col 0 (the canonical PPG amplitude
        # sampled at video fps per src/rppg_sa/data/mcd_rppg.py).
        sync_path = args.ppg_sync_dir / f"{stem}.txt"
        if sync_path.exists():
            arr = np.loadtxt(sync_path, dtype=float)
            ref_waveform = arr[:, 0] if arr.ndim == 2 else arr
            ref_res = windowed_hr_bpm(
                ref_waveform, fs,
                window_seconds=args.window_seconds,
                overlap=args.overlap,
                hr_band=band,
            )
            reference_hr.append(ref_res["hr_bpm"])
            reference_hr_per_window.append(ref_res["hr_bpm_per_window"])
        else:
            reference_hr.append(np.nan)
            reference_hr_per_window.append([])

    joined["extracted_hr_bpm"] = extracted_hr
    joined["reference_hr_bpm"] = reference_hr
    joined["abs_error_bpm"] = (joined["extracted_hr_bpm"] - joined["reference_hr_bpm"]).abs()

    # SNR tertile from manifest (low/mid/high).
    snr_quantiles = joined["mean_snr_db"].quantile([1 / 3, 2 / 3]).tolist()
    def snr_tertile(x: float) -> str:
        if x < snr_quantiles[0]:
            return "low"
        if x < snr_quantiles[1]:
            return "mid"
        return "high"
    joined["snr_tertile"] = joined["mean_snr_db"].apply(snr_tertile)

    cols = [
        "filename", "patient_id", "step", "fps", "n_frames",
        "extracted_hr_bpm", "reference_hr_bpm", "abs_error_bpm",
        "mean_snr_db", "snr_tertile", "pulse", "age", "sex",
    ]
    per_row = joined[cols].copy()
    per_row.to_csv(args.out_dir / "per_row.csv", index=False)

    valid = joined.dropna(subset=["extracted_hr_bpm", "reference_hr_bpm"]).copy()
    n_valid = len(valid)
    if n_valid == 0:
        print("No valid pairs.", file=sys.stderr)
        return 1

    mae = float(valid["abs_error_bpm"].mean())
    rmse = float(np.sqrt(((valid["extracted_hr_bpm"] - valid["reference_hr_bpm"]) ** 2).mean()))
    medae = float(valid["abs_error_bpm"].median())
    r, _ = pearsonr(valid["extracted_hr_bpm"], valid["reference_hr_bpm"])

    summary = {
        "n_videos_total": int(len(manifest)),
        "n_videos_with_pulse_and_reference": int(n_valid),
        "method": "windowed_hr_bpm",
        "window_seconds": args.window_seconds,
        "overlap": args.overlap,
        "hr_band_hz": [args.band_lo_hz, args.band_hi_hz],
        "mae_bpm": round(mae, 3),
        "rmse_bpm": round(rmse, 3),
        "median_abs_error_bpm": round(medae, 3),
        "pearson_r": round(float(r), 4),
        "extracted_hr_mean": round(float(valid["extracted_hr_bpm"].mean()), 2),
        "extracted_hr_std":  round(float(valid["extracted_hr_bpm"].std()), 2),
        "reference_hr_mean": round(float(valid["reference_hr_bpm"].mean()), 2),
        "reference_hr_std":  round(float(valid["reference_hr_bpm"].std()), 2),
    }
    pd.Series(summary).to_json(args.out_dir / "summary.json", indent=2)

    print("\n=== Validation summary (windowed median HR) ===")
    for k, v in summary.items():
        print(f"  {k:35s} {v}")
    print(f"\nReference: MCD-rPPG paper POS baseline = 3.80 bpm MAE on 600 subjects.")
    print(f"Prior in-house result on ~100 videos (mcd_rppg_v2): 12.39 bpm MAE.")
    print(f"This run on {n_valid} videos ({valid['patient_id'].nunique()} subjects): "
          f"MAE = {mae:.2f} bpm.")

    print("\nPer-step (resting/post-exercise) breakdown:")
    by_step = valid.groupby("step").agg(
        n=("filename", "count"),
        mae_bpm=("abs_error_bpm", "mean"),
        ref_hr_mean=("reference_hr_bpm", "mean"),
        ref_hr_max=("reference_hr_bpm", "max"),
    ).round(2)
    print(by_step.to_string())
    by_step.to_csv(args.out_dir / "per_step.csv")

    print("\nPer-SNR-tertile breakdown:")
    by_snr = valid.groupby("snr_tertile").agg(
        n=("filename", "count"),
        mae_bpm=("abs_error_bpm", "mean"),
        snr_min=("mean_snr_db", "min"),
        snr_max=("mean_snr_db", "max"),
    ).round(2).reindex(["low", "mid", "high"])
    print(by_snr.to_string())
    by_snr.to_csv(args.out_dir / "per_snr_tertile.csv")

    # High-HR stratification (>100 bpm reference) — known sub-harmonic-lock
    # regime per src/rppg_sa/extractors/hr.py docstring.
    high_hr = valid[valid["reference_hr_bpm"] > 100.0]
    low_hr = valid[valid["reference_hr_bpm"] <= 100.0]
    print(f"\nHigh-HR (ref > 100 bpm) stratification:")
    print(f"  ref ≤ 100 bpm  n={len(low_hr):3d}  MAE={low_hr['abs_error_bpm'].mean():.2f}")
    print(f"  ref > 100 bpm  n={len(high_hr):3d}  MAE={high_hr['abs_error_bpm'].mean():.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
