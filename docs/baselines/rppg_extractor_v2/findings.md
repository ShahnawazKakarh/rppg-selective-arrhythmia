# rPPG extractor validation on MCD-rPPG (54 subjects, 108 videos)

**Date:** 2026-06-27
**Method:** MediaPipe FaceMesh ROI (forehead + bilateral cheeks) → POS → windowed median HR (10s windows, 50% overlap, 0.7-3.0 Hz band) via `src/rppg_sa/extractors/hr.py::windowed_hr_bpm`. Reference HR: ppg_sync col 0 (per-frame PPG amplitude per `src/rppg_sa/data/mcd_rppg.py::load_ppg_sync`), same windowed-median pipeline applied to the reference waveform.

## Headline

**MAE 11.75 bpm on 108 videos (54 subjects).** Median abs error 6.0 bpm. Pearson r 0.28.

| Reference | MAE bpm | Notes |
|---|---:|---|
| MCD-rPPG paper POS baseline | 3.80 | Full 600 subjects, paper's pipeline |
| Prior in-house (mcd_rppg_v2) | 12.39 | ~100 subjects, same pipeline |
| **This run** | **11.75** | 108 videos, 54 subjects (FullHDwebcam only) |

## Stratifications

- **Step (resting vs post-exercise):** before 11.22 bpm (n=54), after 12.28 bpm (n=54). Marginal step effect.
- **Reference HR:** ≤100 bpm 9.46 bpm (n=98), >100 bpm **34.20 bpm** (n=10). **The high-HR stratum is the dominant error driver** — exactly the sub-harmonic-lock failure mode flagged in `src/rppg_sa/extractors/hr.py`'s docstring.
- **SNR tertile:** low 8.00 / mid 14.33 / high 12.92 bpm. Inverted — likely an artifact of high-HR cases also having high SNR (clean post-exercise pulse) but failing on sub-harmonic lock. Worth jointly stratifying HR × SNR in a follow-up.

## Implications for next paper

The 34 bpm error on the high-HR stratum is the exact failure mode that motivates a learned rPPG extractor. **PhysNet (or similar) is the natural next-paper experiment.** Hypothesis: PhysNet's learned temporal features avoid the sub-harmonic-lock failure mode by encoding rhythm structure beyond the dominant Welch peak.

## Files (gitignored runs/ — local only)

- `runs/extract_mcd_full/manifest.csv` — extraction manifest (108 videos).
- `runs/extract_mcd_full/pulses/*.csv` — POS pulse waveforms.
- `runs/validate_mcd_full/per_row.csv` — per-video extracted vs reference HR + stratifications.
- `runs/validate_mcd_full/summary.json` — aggregate metrics.
- `runs/validate_mcd_full/per_step.csv` — step (resting/post-exercise) breakdown.
- `runs/validate_mcd_full/per_snr_tertile.csv` — SNR tertile breakdown.

## Reproduce

```bash
python scripts/extract_rppg_batch.py \
  --videos-dir data/raw/mcd_rppg/video \
  --out-dir runs/extract_mcd_full \
  --method pos

python scripts/validate_rppg_on_mcd_full.py \
  --manifest runs/extract_mcd_full/manifest.csv \
  --out-dir runs/validate_mcd_full
```

## Limitations

- 54/600 subjects (9%) — local subset only. Full 600 would require downloading the gated HF mirror (~135 GB).
- FullHDwebcam camera only. USBVideo and IriunWebcam cameras add 2× more videos per subject but are not downloaded locally.
- Single fps measurement per video; if MediaPipe's fps differs from the dataset's nominal 30 Hz, HR is biased proportionally. Spot-checks in `manifest.csv` show fps in {24.0, 29.9, 30.0} across videos — variation is small but present.
- No bandpass pre-filter on either signal; rely on Welch's robustness to broadband noise. Adding a 0.7-4 Hz Butterworth bandpass before HR estimation is a likely small improvement.
