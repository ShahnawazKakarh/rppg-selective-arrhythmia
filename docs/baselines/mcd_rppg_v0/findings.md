# MCD-rPPG v0 — Classical Extractor Baseline

**Date:** 2026-06-17
**Subset:** 8 records (4 subjects × resting + post-exercise) from the
`milai-oks-sakura/mcd_rppg` Hugging Face mirror, frontal-webcam view only.

## Result

| Method | MAE vs PPG-sync GT (bpm) | Notes |
|---|---|---|
| **POS** (Wang et al. 2017) | **5.35** | Paper baseline on full dataset: 3.80 bpm |
| CHROM (de Haan & Jeanne 2013) | 20.08 | Consistent sub-harmonic lock-on |

Face detection (MediaPipe FaceMesh): **100% across all 8 clips.**

## Per-record numbers

```
patient  session  fps    PPG-sync   biomarker   CHROM       POS
1020     after    29.90  77.09      100         49.05 (-28) 77.09 ( +0)
1020     before   29.90  77.09       83         52.56 (-25) 77.09 ( +0)
1024     after    29.90  71.83       88         47.30 (-25) 50.81 (-21)
1024     before   24.00  60.47       78         53.44 ( -7) 57.66 ( -3)
1035     after    29.90  73.58       93         52.56 (-21) 61.32 (-12)
1035     before   29.90  68.33       83         50.81 (-18) 63.07 ( -5)
1091     after    24.00  80.16      132         57.66 (-23) 81.56 ( +1)
1091     before   24.00  67.50       94         52.03 (-15) 67.50 ( +0)
```

## Two ground-truth references

The dataset provides three potential HR references; only one is appropriate
for scoring rPPG over the video window:

1. **`ppg_sync/*.txt` column 1** — contact-PPG amplitude resampled to the
   video frame rate. **One row per video frame.** This is the genuine
   ground truth synchronized to the video time window. We use it as the
   primary GT.
2. **`db.csv` `pulse` column** — discrete clinical cuff reading. POS error
   against this is 26.86 bpm. We attribute this to cuff-vs-video timing
   mismatch (e.g. subject 1091: biomarker 132 bpm but PPG-sync says 80 bpm
   during the actual 3-minute video). Kept as a reference column for
   debugging only.
3. **`ecg/*.json`** — multi-lead clinical ECG. Not used yet; will be the
   gold reference for the eventual AF classification task.

## ppg_sync column format (confirmed)

We initially mis-interpreted `ppg_sync/*.txt`. After probing
(`scripts/probe_ppg_sync.py`):

- Each file has **exactly one row per video frame**, not per 100 Hz PPG sample.
- **Column 1** = contact-PPG amplitude, 8-bit unsigned (0–255), resampled to
  video frame rate (~30 Hz; sometimes 24 Hz).
- **Column 2** = sync-offset metadata (timing offset between the underlying
  100 Hz PPG sample and the video frame). It is *not* inter-sample delta
  time; its values sum to far less than the recording duration.

This means the PPG signal we score against is uniformly sampled at the same
rate as the video, which makes alignment trivial (no resampling needed).

## Pipeline notes

- MediaPipe FaceMesh (0.10.18) is pinned for `solutions.face_mesh` API
  compatibility. 0.11+ removed the legacy `mp.solutions` lazy-attribute
  layout.
- OpenCV's `CAP_PROP_FRAME_COUNT` is unreliable on these AVI/MPEG-4 files
  (returns wrong values). We accumulate frames dynamically instead of
  pre-allocating.
- All 8 clips show `[mpeg4] Error at MB:` warnings from FFmpeg during decode.
  These are non-fatal but may explain some of the CHROM sub-harmonic lock-on
  (CHROM is more sensitive to high-frequency artifacts than POS). Switching
  to PyAV / imageio-ffmpeg may improve CHROM; deferred until we determine
  whether CHROM is on the methodological critical path.

## Implementation bugs fixed during this baseline

1. **POS sign error** (`src/rppg_sa/extractors/pos.py`): had
   `h = S[0] + alpha * S[1]`; canonical POS is
   `h = S[0] - alpha * S[1]` (Wang et al. 2017, Eq. 11). The `+` variant
   was contributing dominant noise from the wrong-sign chrominance band.
2. **POS overlap-add normalization**: matched CHROM's Hanning window +
   counts-array normalization. Without it, summed window contributions
   accumulate without proper averaging.
3. **MCD-rPPG loader `iter_samples`**: blanket `float()` of all biomarker
   columns crashed on the `sex` field ('F'/'M' string); switched to
   try/except per cell.
4. **Face ROI extraction**: switched from preallocation-by-frame-count to
   dynamic accumulation; switched to explicit
   `from mediapipe.solutions import face_mesh` import path.

## What this baseline does NOT yet show

- Generalization across all 600 subjects. n=8 is sufficient to demonstrate
  pipeline correctness; n=600 is needed for a publishable comparison.
- The other two camera views (USBVideo, IriunWebcam) and three head angles
  (front, left, right). Frontal-webcam is the easiest case.
- Windowed (rather than full-clip) HR estimation. Welch on a single 3-minute
  clip averages out within-clip HR variation. Per-10-second-window HR will
  matter for the eventual selective prediction work, where we need
  per-segment confidence rather than per-clip.

## Next steps

1. Pull a 50-subject subset and re-run; expect POS MAE to drop toward the
   paper's 3.80 bpm as cohort grows.
2. Implement per-window HR estimation (~10s windows, 50% overlap) for
   per-segment confidence inputs to the eventual selective-prediction work.
3. Investigate CHROM sub-harmonic lock — likely fixable by widening the
   ROI (add chin) or detrending before chrominance projection.
4. Then: pivot to the AF classification critical path (this rPPG work is
   the *substrate* for selective prediction, not the contribution).
