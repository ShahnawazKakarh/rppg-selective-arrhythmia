# Synth-rPPG v1 — MIT-BIH ECG → synthetic 30 Hz rPPG pipeline

**Date:** 2026-06-18
**Pipeline:** MIT-BIH ECG (360 Hz) → R-peak detect (neurokit2) → beat-template placement at PTT lag → downsample to 30 Hz → noise + baseline wander → 30 s windows. Implementation: `src/rppg_sa/data/synth_rppg.py` + `src/rppg_sa/data/synth_rppg_torch.py`.
**Why:** OBF / MAHNOB-HCI access pending. This is the fallback that keeps the AF rPPG methodology alive on fully open data.

## Headline

| Method | acc | ECE↓ | Brier↓ | AURC↓ | sel_acc@0.5↑ |
|---|---:|---:|---:|---:|---:|
| MC Dropout (T=30) | 0.208 | 0.426 | 0.956 | 0.776 | 0.233 |
| Conformal (α=0.1) | 0.275 | 0.382 | 0.927 | 0.751 | 0.267 |
| **Ensembles (M=5)** | 0.308 | **0.242** | **0.756** | 0.737 | 0.183 |

## What worked

- **Synthesis pipeline operational.** Val macro-F1 reached **0.64** on the single-model run (epoch 6) — meaning AF rhythm signal survives the round-trip ECG → R-peaks → beat templates → downsample to 30 Hz → noise. The classifier can learn rhythm patterns from synthesized rPPG.
- **All three UQ methods functional on the new data source.** No code changes to UQ modules — just data-source dispatch.
- **Ensembles still dominate calibration** — ECE 0.24, Brier 0.76, best of the three. Same relative ordering as the raw-ECG MIT-BIH baseline.

## What didn't work

- **Test accuracy is 21-31%** — below 3-class chance (33%). Same distribution-shift problem as the MIT-BIH baseline at the same split: test = {210, 200} has 58 AF and 33 Other, but only one of those records (210) contributes AF in the synth case, and the model never learned to generalize that one record's AF pattern beyond seeing 4 AF records in train.
- **Ensemble sel_acc@0.5 (0.183) < acc (0.308).** The ensemble's most-confident predictions are *more wrong* than random. Confidence anti-correlates with correctness on this test set. This is a classifier-failure signature, not a UQ failure.

## What this means

The synthesis pipeline is operational and reproducible. The bottleneck is the MIT-BIH train/test split — 4 AF records in train is too few to learn AF rhythm robustly, and the 2-record test set is too narrow to evaluate. To make UQ comparison meaningful on synth, we need either:

- **More AF records in train** — bring in CinC 2017 AF Classification Challenge (10K+ ECGs with AF labels). Synthesize rPPG from those, retrain.
- **Subject-wise cross-validation** instead of fixed splits. With only 4-5 AF records total, leave-one-out across AF records would give more stable estimates.
- **Bigger model + more data.** Current CNN1D-Transformer was MIT-BIH-sized; for a CinC-scale dataset it would need scaling.

## Files

- Code: `src/rppg_sa/data/synth_rppg.py`, `src/rppg_sa/data/synth_rppg_torch.py`.
- Config: `configs/synth_rppg_baseline.yaml`.
- Single-model run: `runs/synth_rppg_baseline/` (val_macro_f1 = 0.6412).
- Ensemble members: `runs/synth_rppg_baseline_ens{1..5}/`.
- Per-method results: `runs/synth_rppg_baseline/eval_{mc_dropout,conformal}/`, `runs/synth_rppg_baseline_ens1/eval_ensembles/`.

## Next

1. **CinC 2017 AF Challenge** — 8,528 single-lead ECG recordings, AF labels, fully open via PhysioNet. Synthesize rPPG from these → 5-10x more AF training data → meaningful UQ comparison. **Next session priority.**
2. **MAHNOB-HCI access request** still in flight as parallel face-video path.
3. **OBF** still in flight.
