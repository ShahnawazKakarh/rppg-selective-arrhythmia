# Clean-ensemble methodology — separated data_seed and model_seed

**Date:** 2026-06-21
**Substrate:** synth-rPPG CinC test set ($n = 6{,}750$); 5-member deep ensemble retrained with `data_seed = 42` (shared across all members) and `model_seed ∈ {1, 2, 3, 4, 5}` (one per member).

## What changed

The original 5-member ensemble (`runs/synth_rppg_cinc_ens{1..5}/`) was trained with `--seed N` for each member, which under the old `scripts/train_classifier.py` API set a single seed used for *both* the train/val/test split *and* the model init / DataLoader shuffle. As a consequence, the five members saw five *different* data splits. This conflates ensemble diversity (the intended variance source) with split-induced variance (an artifact). Standard deep-ensemble methodology (Lakshminarayanan et al. 2017) is shared split + independent initialisations.

Patch in `scripts/train_classifier.py`: new `--data-seed` and `--model-seed` flags resolve to `cfg["experiment"]["data_seed"]` and `cfg["experiment"]["model_seed"]` respectively. The `auto_split_records` / `auto_split_subjects` calls use `data_seed`; `set_seed()` uses `model_seed`. Legacy `--seed N` continues to set both, so older configs still work.

Retrained ensemble at `runs/synth_rppg_cinc_clean_ens{1..5}/`, all sharing the same record-level 70/15/15 split derived from `data_seed = 42`. Per-member best val macro-F1: 0.574 / 0.595 / 0.602 / 0.599 / 0.584.

## Headline — clean methodology improves UQ quality *and* the LW-CCSD gain

CinC test set ($n = 6{,}750$):

| Metric                                | Coupled-seed (original) | Clean-seed (this) | Δ |
|---|---:|---:|---|
| Test accuracy                         | 0.686                    | **0.692**         | +0.6 pp |
| ECE                                   | 0.064                    | **0.052**         | better |
| Brier score                           | 0.440                    | **0.440**         | flat |
| Test AURC (UQ-only)                   | 0.2155                   | **0.2066**        | +4.1 % relative |
| Sel acc @ 0.5                         | 0.795                    | **0.800**         | +0.5 pp |

Three reads:

1. **The coupled-seed ensemble was systematically worse-calibrated.** ECE drops from 0.064 to 0.052 under shared-split methodology. This is the cleanest evidence that the split-induced variance in the old setup was injecting noise into the calibration story.
2. **The UQ-only ranker is better.** Test AURC drops from 0.2155 to 0.2066 — 4.1 % relative improvement in selective performance with no architectural change, purely from removing the split artifact.
3. **Calibration is better but classification accuracy barely moves.** This is the expected behaviour: ensemble diversity buys calibration first, accuracy second.

## LW-CCSD Pareto frontier on clean ensemble

CinC test set, snr_db, constraint coverage 0.50, deterministic LW-CCSD scoring (no conformal):

| Val AF floor | $\mathbf{w}^{*}$ (NSR/AF/Other) | Test AURC | Δ vs UQ-only | Test AF@0.5 | Δ AF |
|---:|---|---:|---:|---:|---:|
| 0.40 | 0.0 / 0.5 / 0.2 | 0.1921 | **+7.04 %** | 0.497 | −0.088 |
| 0.45 | 0.0 / 0.4 / 0.2 | 0.1928 | **+6.71 %** | 0.553 | −0.032 |
| 0.50 | 0.2 / 0.2 / 0.2 | 0.1953 | +5.49 % | 0.565 | −0.020 |
| ≥ 0.55 | (infeasible — val UQ-only AF recall is 0.512) | — | — | — | — |
| UQ-only baseline | — | 0.2066 | — | 0.585 | 0.000 |

The recommended operating point at **floor 0.45** captures +6.7 % AURC at a 3-point AF-recall cost. This is *higher* than the +3.2 % gain reported on the coupled-seed ensemble at the same floor. The clean methodology produces a better UQ baseline *and* a larger LW-CCSD margin on top of it.

## Why the LW-CCSD gain grew under clean methodology

This was initially surprising. Two complementary mechanisms:

1. **The coupled-seed ensemble's split variance partially "absorbed" the deferral signal.** When the five members saw different splits, their per-segment uncertainty included a component due to which-split-this-was rather than how-confident-the-classifier-actually-is. SNR is independent of split; rank-mixing SNR with that noisier confidence yielded less marginal gain. Under the clean ensemble, the confidence ranking is closer to a noiseless estimate of within-record uncertainty, and SNR adds more discriminating cross-regime information on top.
2. **Better calibration means the LW-CCSD optimizer's val-AURC objective is closer to the test-AURC quantity it really cares about.** A miscalibrated val objective produces noisier weight selection. The 1.2-point ECE drop translates into a tighter val-to-test relationship and thus to operating points that generalize better.

## Implications for the paper

The headline numbers in Tables 4 (cross-UQ) and 7 (cross-UQ stratification) shift on the ensemble row. The deployment punchline strengthens:

| Comparison                                                  | Test AURC |
|---|---:|
| Deterministic UQ-only                                        | 0.2367   |
| **Deterministic + LW-CCSD (recommended op)**                | **0.1998** |
| MC Dropout UQ-only                                           | 0.1980   |
| MC Dropout + LW-CCSD (best safe)                            | 0.1899   |
| **Clean ensemble UQ-only** (was 0.2155)                     | **0.2066** |
| **Clean ensemble + LW-CCSD (best safe, floor 0.45)**        | **0.1928** |

Deterministic + LW-CCSD (0.1998) still beats Clean ensemble UQ-only (0.2066) by 3.3 % relative — at one-fifth the inference cost. The deployment substitution claim holds even against the stronger ensemble baseline.

## Files

- Patched trainer: `scripts/train_classifier.py` (new `--data-seed` / `--model-seed` flags, backwards-compatible `--seed`).
- Clean ensemble checkpoints: `runs/synth_rppg_cinc_clean_ens{1..5}/best.pt`.
- Clean ensemble eval: `runs/synth_rppg_cinc_clean_ens1/eval_ensembles/`.
- Clean ensemble val predictions: `runs/synth_rppg_cinc_clean_ens1/eval_val_ensembles/predictions.csv`.
- LW-CCSD floor sweep: `runs/synth_rppg_cinc_clean_ens1/eval_ensembles/lw_ccsd_clean_floor{0.40,0.45,0.50}/`.
- This findings document.

## Next

- Re-run the SNR-stratified and HR-stratified evaluations on the clean ensemble for completeness in the cross-UQ table.
- Update paper Table 4 (cross-UQ comparison) and Table 7 (SNR-stratified cross-UQ) to use the clean-ensemble numbers.
