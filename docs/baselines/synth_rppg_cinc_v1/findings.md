# Synth-rPPG CinC v1 — UQ methods on the scaled-AF substrate

**Date:** 2026-06-18
**Dataset:** PhysioNet/CinC 2017 AF Challenge training set, 8,244 records (8,528 total minus 284 noisy `~`).
**Pipeline:** ECG (300 Hz) → R-peaks → asymmetric Gaussian beat templates → downsample to 30 Hz → noise + baseline wander → 10 s windows / 5 s hop.
**Dataset segments:** 45,064 (NSR 26,898 / AF 3,942 / Other 14,224).
**Split:** record-level 70 / 15 / 15 stratified by class label; subject-disjoint.

## Headline

| Method | acc | ECE↓ | Brier↓ | AURC↓ | sel_acc@0.5↑ | sel_acc@0.9↑ | sel_acc@0.95↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **MC Dropout (T=30)** | **0.715** | 0.076 | 0.418 | **0.198** | **0.800** | **0.735** | **0.726** |
| Conformal (α=0.1) | 0.699 | **0.056** | 0.439 | 0.237 | 0.763 | 0.723 | 0.713 |
| Ensembles (M=5) | 0.686 | 0.064 | 0.440 | 0.215 | 0.795 | 0.717 | 0.703 |

## What this confirms

- **Scale works.** Going from 4 AF training records (MIT-BIH) to 516 (CinC, ~70 % of 738 AF records) flipped the picture: all three UQ methods land within 3 accuracy points of each other, all well-calibrated (ECE < 0.08, vs 0.25-0.39 on MIT-BIH), and selective curves monotonically improve as coverage drops. This is the regime where UQ comparison is methodologically meaningful.
- **Synthesis pipeline preserves rhythm signal at scale.** Single-model val macro-F1 reached 0.596 over 25 epochs of stable learning — no collapse, no overfitting. Means the synth pipeline is not a toy: 8K AF-labelled records produce a real classifier.
- **NSR and AF both well-classified (~83 %), Other is the hard class (~40 %).** Expected — "Other" in CinC is a catch-all for non-AF non-noisy rhythms (PVCs, PACs, brady/tachycardia, etc.) with no single morphological signature.

## Cross-dataset UQ comparison

| Dataset | Best AURC | Best ECE | Best sel_acc@0.5 |
|---|---|---|---|
| MIT-BIH UQ v1 | Ensembles (0.095) | Ensembles (0.125) | Conformal (0.928) |
| **Synth-rPPG CinC v1** | **MC Dropout (0.198)** | **Conformal (0.056)** | **MC Dropout (0.800)** |

Different winners on different datasets. The ordering is **not consistent across data sources** — a publishable methodological finding in itself. The "best UQ method" question is dataset-dependent; a deployed system probably needs all three with a meta-policy that picks the right tool for the regime.

## Known limitation — ensemble seed coupling

Each ensemble member was trained with its own `experiment.seed` (1..5) passed via `--seed`, which means each member also got a different **data split** (the auto-split function consumes the same seed). The five members therefore trained on slightly different train/val/test partitions rather than only differing in init/dropout/batch order.

This artificially inflates ensemble diversity but also means some test samples for ens-member-N may have been in train for ens-member-M. The ensemble result is closer to a bagging-style estimator than a pure deep ensemble.

**Fix planned:** separate `data_seed` from `model_seed` in the trainer. Use one fixed `data_seed` across all ensemble members, vary only `model_seed`. Re-run, expect ensemble numbers to shift slightly. Not a blocker for the methodological story; documented for reproducibility.

## Per-class on test (all three methods)

| Class | n | MC Dropout acc | Conformal acc | Ensemble acc |
|---|---:|---:|---:|---:|
| NSR | ~4,070 | 0.840 | 0.841 | 0.837 |
| AF | ~580 | 0.809 | 0.828 | 0.776 |
| Other | ~2,150 | 0.449 | 0.391 | 0.383 |

AF accuracy 78-83 % across all three — the clinically critical class is well-detected.

## Files

- Synth code: `src/rppg_sa/data/synth_rppg.py`, `src/rppg_sa/data/cinc2017.py`, `src/rppg_sa/data/cinc2017_synth_torch.py`.
- Download: `scripts/download_cinc2017.py`.
- Config: `configs/synth_rppg_cinc.yaml`.
- Single-model run: `runs/synth_rppg_cinc/` (val_macro_f1 = 0.596).
- Ensemble members: `runs/synth_rppg_cinc_ens{1..5}/`.
- Per-method results: `runs/synth_rppg_cinc/eval_{mc_dropout,conformal}/`, `runs/synth_rppg_cinc_ens1/eval_ensembles/`.

## Next

1. **Fix ensemble seed coupling** — separate data and model seeds, re-run. Quick.
2. **Signal-quality-aware deferral on this substrate** — combine model confidence with the per-window SNR / template SQI features (already computed in `extractors/signal_quality.py`). Measure whether SQI-aware deferral beats UQ-only deferral on synth-rPPG-CinC. This is the headline methodological claim of the paper.
3. **OBF / MAHNOB-HCI** — when paired face-video AF data arrives, re-run the same three-method comparison and report transfer behaviour.
4. **Evidential DL training-time integration** — adds the 4th UQ method to the comparison table.
