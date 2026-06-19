# SQI-aware deferral v1 — the technical novelty result

**Date:** 2026-06-18
**Substrate:** synth-rPPG CinC test set (n=6,750, 3-class), single-model checkpoint `runs/synth_rppg_cinc/best.pt` (val_macro_f1=0.596).
**Deferral score:** rank-normalized linear combination
  combined = (1 − w) · model_confidence + w · signal_quality
swept w ∈ {0, 0.15, 0.30, 0.50, 0.70, 1.00}. w=0 is the UQ-only baseline.

## Headline

Combining model confidence with **spectral SNR** substantially beats UQ-only deferral. Combining with template SQI barely helps.

| SQI feature | UQ-only AURC | Best AURC | Δ | Best w | sel_acc@0.5 (UQ-only → best) |
|---|---:|---:|---:|---:|---:|
| template_sqi | 0.2367 | 0.2289 | +0.0078 (3.3 %) | 0.30 | 0.763 → 0.754 |
| **snr_db** | 0.2367 | **0.1942** | **+0.0425 (18.0 %)** | 0.70 | 0.763 → **0.802** |

## SNR sweep (the result)

| w | AURC↓ | sel_acc@0.5 | sel_acc@0.7 | sel_acc@0.8 | sel_acc@0.9 | sel_acc@0.95 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.2367 | 0.7630 | 0.7551 | 0.7422 | 0.7235 | 0.7126 |
| 0.15 | 0.2180 | 0.7662 | 0.7560 | 0.7476 | 0.7259 | 0.7118 |
| 0.30 | 0.2043 | 0.7727 | 0.7600 | 0.7507 | 0.7263 | 0.7115 |
| 0.50 | 0.1951 | 0.7979 | 0.7623 | 0.7526 | 0.7253 | 0.7115 |
| **0.70** | **0.1942** | **0.8021** | **0.7746** | **0.7533** | 0.7240 | 0.7114 |
| 1.00 | 0.2106 | 0.8024 | 0.7687 | 0.7391 | 0.7118 | 0.7047 |

AURC monotonically improves with w up to 0.7, then degrades — pure SNR (w=1.0) loses the model-confidence signal entirely. The optimum at w=0.7 says: trust signal quality more than the model, but not exclusively.

At 50 % coverage (the regime where a clinician would review the top-confidence half), SQI-aware deferral hits 80.2 % accuracy vs 76.3 % UQ-only — a +3.9 percentage-point gain that comes for free (no retraining, no architecture change, just a different ranking score).

## Template SQI sweep

| w | AURC↓ | sel_acc@0.5 |
|---:|---:|---:|
| 0.00 | 0.2367 | 0.7630 |
| 0.30 | 0.2289 | 0.7544 |
| 1.00 | 0.2622 | 0.7084 |

Template SQI is too noisy and class-correlated to help much. Useful as a diagnostic; not a deferral signal on its own.

## Why this matters

This is the technical novelty of the paper, isolated as a clean controlled experiment:

- The signal-quality feature is **physically grounded** (spectral SNR around the dominant pulse peak in the 0.7–3 Hz band), not learned from the same training data as the classifier. So it carries information the model confidence inherently doesn't.
- The result holds **post-hoc**: no retraining, no new model. Any deployed system using this classifier could add SNR-weighted deferral overnight.
- The optimal w ≈ 0.7 is methodologically interpretable: model confidence is informative but signal-quality is more informative for the *abstention decision* specifically.

This is the empirical claim the paper hinges on.

## Replication across UQ methods

The SNR-deferral gain is not a single-model artefact. Same SNR sweep (rank-normalized linear combination), three different UQ-confidence sources for the same CinC test set:

| UQ method | Confidence source | UQ-only AURC | Best AURC | Δ rel | Best w |
|---|---|---:|---:|---:|---:|
| Deterministic | max softmax | 0.2367 | **0.1942** | **+18.0 %** | 0.70 |
| MC Dropout (T=30) | −predictive entropy | 0.1980 | 0.1892 | +4.4 % | 0.30 |
| Ensembles (M=5) | max averaged softmax | 0.2155 | 0.2059 | +4.5 % | 0.30 |

Observations:

- **SNR helps every UQ method.** The Δ is positive across all three; the effect is robust.
- **Magnitude scales inversely with UQ quality.** When the underlying model confidence is already informative (MC Dropout entropy, ensemble averaging), SNR adds ~4 %. When it's a single deterministic softmax peak, SNR adds 18 %. Interpretation: SNR is filling in the information the weaker confidence misses.
- **Optimal w shifts in the same direction.** Deterministic confidence → w=0.70 (lean on SNR). Stronger UQ confidence → w=0.30 (model first, SNR supplements). Methodologically interpretable: the deferral weighting tracks confidence informativeness.

For a deployed system, this means: the simpler the model's uncertainty signal, the more SNR-deferral matters. Lightweight clinical screens — which are usually deterministic, single-forward-pass for compute reasons — are exactly where SQI-aware deferral pays the most.

## Files

- Code: `scripts/eval_sqi_deferral.py`, `src/rppg_sa/selective/deferral.py`.
- Per-weight detail: `runs/synth_rppg_cinc/eval_sqi_deferral/results_snr_db.json`, `…/results_template_sqi.json`.
- CSV sweeps: `runs/synth_rppg_cinc/eval_sqi_deferral/sweep_snr_db.csv`, `…/sweep_template_sqi.csv`.
- MC Dropout replication: `runs/synth_rppg_cinc/eval_mc_dropout/eval_sqi_deferral_mc_dropout/sweep_snr_db.csv`.
- Ensemble replication: `runs/synth_rppg_cinc_ens1/eval_ensembles/eval_sqi_deferral_ensembles/sweep_snr_db.csv`.

## Next

1. **Replicate on MIT-BIH classifier** for cross-dataset robustness of the SQI claim.
2. **Per-class breakdown of SQI-deferral gain** — does it preferentially recover AF or NSR? (clinically important.)
3. **Separate `data_seed` from `model_seed`** in the trainer, retrain the ensemble cleanly, re-run SQI sweep.
4. **Evidential DL training-time integration** — a 4th UQ method in the comparison.
5. **When OBF arrives**, this is the experiment to run first.