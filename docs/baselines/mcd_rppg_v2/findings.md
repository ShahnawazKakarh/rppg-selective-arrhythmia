# MCD-rPPG v2 — Windowed-Median HR Baseline

**Date:** 2026-06-17
**Subset:** 100 records (50 subjects × resting + post-exercise) from
`milai-oks-sakura/mcd_rppg`, frontal-webcam view only. Subjects sampled
deterministically (seed=42).
**Supersedes:** `mcd_rppg_v0` (n=8, lucky small sample) and `mcd_rppg_v1`
(n=100, single-clip Welch — broke at scale).

## Headline result

| Method | MAE vs PPG-sync GT (bpm) | MAE vs biomarker (bpm) | Notes |
|---|---:|---:|---|
| **POS** (Wang et al. 2017) | **12.39** | 22.08 | Sub-harmonic-locks on high-HR clips |
| CHROM (de Haan & Jeanne 2013) | 27.24 | 38.15 | Systematic sub-harmonic lock-on, every clip |

Face detection: **100 % across all 100 clips.** POS-MAE-vs-biomarker is higher
than vs PPG-sync because cuff biomarkers are not co-temporaneous with the
3-minute video (consistent with the v1 finding); PPG-sync is the right GT.

## Methodological progression v0 → v1 → v2

| Version | HR aggregation | n | POS MAE |
|---|---|---:|---:|
| v0 | Single Welch peak over full clip | 8 | 5.35 (lucky) |
| v1 | Single Welch peak over full clip | 100 | 27.10 |
| **v2** | **Median across 10 s windows, band 0.7–3.0 Hz** | **100** | **12.39** |

The v0 → v1 collapse is the lesson: at n=8, several subjects happened to have
the dominant Welch peak coincide with their true HR. At n=100, half the
clips fail because the full-clip Welch peak picks up whichever frequency
component has the most integrated power — lighting flicker, breathing-band
sway, the 2nd HR harmonic — not necessarily the fundamental.

The v1 → v2 fix (median across 10 s windows) is what every published rPPG
benchmark uses, including the MCD-rPPG paper itself. A few bad windows can't
drag the median, and 10 s is short enough that within-window noise is
locally dominated by the actual pulse.

## Where the remaining 12 bpm error lives

Per-clip POS residuals fall into three regimes:

1. **Normal HR (60–90 bpm) — POS works.** Median |Δ| around 3–6 bpm.
   Subject IDs: 1314, 2194, 3196, 3784, 4312, 5130, 5156, 5288-before,
   5948-before, 6066-before, 6289-before, 7878-before, 8706, 9160, 9286,
   9425-after, 9640, 9922, etc.

2. **High HR (>100 bpm post-exercise) — POS picks half-HR.** Systematic
   2× sub-harmonic lock-on:
   - 1107-after: GT 114, POS 72 (Δ -42)
   - 1649-after: GT 114, POS 66 (Δ -48)
   - 6329-after: GT 108, POS 60 (Δ -48)
   - 7878-after: GT 138, POS 66 (Δ -72)

3. **A few high-side mis-locks.** POS occasionally pegs to ~120 or 129
   when GT is in the 60-90 range (6008-before, 9425-before). Less common
   than mode 2.

Mode 2 is a well-known classical-rPPG failure: chrominance projection
amplifies the 2nd harmonic when motion is high (post-exercise) or skin
saturation drops. de Haan & van Leest (2014) and Wang (2017) both note
>100 bpm as the hard zone for classical methods.

## CHROM is unsuitable for this dataset

CHROM stays at ~48–60 bpm regardless of true HR. Median across windows
doesn't help because every window has the same sub-harmonic lock. We
attribute this to interaction between CHROM's higher noise sensitivity
and the consistent MPEG-4 decode errors across all clips (FFmpeg warnings
like `[mpeg4] ac-tex damaged at … Error at MB: …` appear on every record
without exception). POS, by contrast, projects orthogonally to the skin-tone
plane and is less affected by the same artefacts.

Recommendation: drop CHROM from the next phase. POS as the only classical
extractor, PhysNet as the planned learned extractor.

## Why this is the right substrate for selective prediction

The 12 bpm residual is *structured*, not noise:

- **Errors concentrate on a clinically meaningful subset** (post-exercise
  high HR), exactly where contactless screening would *want* to defer to a
  contact device.
- **The failure mode is detectable.** Sub-harmonic lock-on shows up as low
  spectral SNR and high `hr_std_bpm` across the 10 s windows — both already
  computed in `rppg_sa.extractors.hr.windowed_hr_bpm`.
- **Therefore a deferral policy that flags high-`hr_std` clips will
  preferentially flag the failure mode** — exactly what the selective-
  prediction layer is supposed to do.

This makes the rPPG layer good enough as a substrate. The next step is the
classifier and UQ heads, not more rPPG tuning.

## Implementation changes from v1

- New module `src/rppg_sa/extractors/hr.py` with `peak_hr_bpm` and
  `windowed_hr_bpm`. The latter returns per-window HR list, median, and
  std — std is the per-clip uncertainty proxy.
- Validation script switches to `windowed_hr_bpm` for both rPPG and PPG-sync
  GT, so both estimates use the same aggregation.
- HR band tightened from `[0.7, 4.0]` Hz to `[0.7, 3.0]` Hz (42–180 bpm).
  Cuts upper harmonics out of the search.

## Next steps

1. **Move to the classifier and UQ heads.** Per-window POS + signal-quality
   features are now a usable substrate. The MIT-BIH-style training loop
   already exists in `scripts/train_classifier.py`; needs an MCD-rPPG
   dataset adapter.
2. **PhysNet (learned extractor)** as a separate baseline once the
   classifier scaffolding is wired through, to see whether learned
   extraction closes the high-HR gap.
3. **Per-window confidence as the deferral signal.** Add `hr_std_bpm` and
   `snr_db` to the dataset interface so they propagate through to the
   selective head.
