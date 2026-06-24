# Evidential Deep Learning (EDL) — 4th UQ method

**Date:** 2026-06-24
**Substrate:** synth-rPPG CinC test set ($n = 6{,}750$); EDL classifier trained with `data_seed = 42` (shared with all other UQ checkpoints) and `model_seed = 1`. Architecture: same CNN1D + Transformer backbone, softmax head replaced by an Evidential (Dirichlet) head (`src/rppg_sa/models/edl_classifier.py`, `src/rppg_sa/uncertainty/evidential.py`). Loss: EDL Type-II MSE with KL annealing over 10 epochs (Sensoy et al. 2018).

## Headline — EDL is the strongest UQ-only baseline, and LW-CCSD does not help on top

| UQ Method                                       | Test Acc | ECE     | UQ-only AURC | Best safe LW-CCSD | Δ AURC      | AF cost |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic                                    | 0.686    | —       | 0.2367       | 0.1998             | **+15.6 %** | −0.081 |
| MC Dropout ($T = 30$)                            | 0.715    | 0.076   | 0.1980       | 0.1899             | +4.1 %      | −0.029 |
| Clean Ensemble ($M = 5$)                         | 0.692    | 0.052   | 0.2066       | 0.1928             | +6.7 %      | −0.032 |
| **EDL** (this checkpoint)                        | **0.739** | 0.137  | **0.1725**   | 0.1764             | **−2.3 %**  | −0.013 |

Three observations:

1. **EDL has the best UQ-only AURC and the best test accuracy of all four methods.** Test accuracy 0.739 (vs MC Dropout 0.715, Clean Ensemble 0.692, Deterministic 0.686) and test AURC 0.1725 (best of all). This is a 16.6 % relative AURC improvement over the clean ensemble at one-fifth the inference compute — and a 27.1 % improvement over the deterministic single-pass baseline.

2. **EDL is the only UQ source studied where LW-CCSD does not help.** Every feasible LW-CCSD operating point produces a *negative* ΔAURC vs UQ-only. The best result (floor 0.30, $\mathbf{w}^{*} = (0.0, 0.3, 0.4)$) gives test AURC 0.1764 — a 2.3 % relative regression vs UQ-only.

3. **EDL is the worst-calibrated of the four methods.** ECE 0.137 (vs 0.052–0.076 for the others). This is consistent with the EDL literature: the Dirichlet evidence head trades calibration sharpness for ranking quality and explicit-uncertainty modelling. The high ECE does *not* translate into poor AURC — quite the opposite.

## EDL LW-CCSD Pareto frontier

Val UQ-only AF recall at $c_0 = 0.50$ is only 0.313 (EDL is conservative on AF), so floors above 0.35 are infeasible.

| Val AF floor | $\mathbf{w}^{*}$ (NSR/AF/Other) | Test AURC | Δ vs UQ | Test AF@0.5 | Δ AF      |
|---:|---|---:|---:|---:|---:|
| 0.20         | 0.0 / 0.5 / 0.4                 | 0.1777    | −3.02 % | 0.331       | −0.068    |
| 0.25         | 0.0 / 0.4 / 0.4                 | 0.1770    | −2.65 % | 0.367       | −0.032    |
| 0.28         | 0.0 / 0.4 / 0.4                 | 0.1770    | −2.65 % | 0.367       | −0.032    |
| **0.30**     | **0.0 / 0.3 / 0.4**             | **0.1764** | **−2.28 %** | **0.386** | **−0.013** |
| UQ-only      | —                               | 0.1725    | —       | 0.399       | 0.000     |
| ≥ 0.35       | (infeasible — val UQ-only AF recall is 0.313) | — | — | — | — |

The Pareto frontier is *inverted* relative to the other three UQ sources: every LW-CCSD operating point sits *above* UQ-only on test AURC. The deferral rule does not improve the selective ranking — it can only relax the val AF-recall constraint at increasing test cost. This is what a redundant auxiliary feature looks like.

## Mechanism — why EDL absorbs the SQI signal

In the EDL formulation each predicted class receives a non-negative evidence score $e_c$, and Dirichlet concentrations are $\alpha_c = e_c + 1$. Total evidence $S = \sum_c \alpha_c$ and Dirichlet uncertainty $u = K / S$. The training objective penalises the model for confidently predicting wrong classes and also for being over-confident on any class when total evidence is low.

A noisy signal is hard to classify with high evidence: there is no rhythm template, no harmonic structure, no consistent beat morphology. Under EDL training, such a signal produces *low evidence* on every class, hence *low total $\alpha$*, hence *high $u$*. This is exactly the same content as a low SNR score: both quantities are large precisely when the input lacks a clean cardiac signal.

In other words, EDL has *learned* what spectral SNR provides externally. Mixing the two ranks is therefore double-counting the same information, and the noise component of the externally-computed SNR rank now reduces selective quality rather than adding to it.

The other three UQ sources do not have this property:
- **Deterministic softmax** ignores signal quality entirely; it computes the max-class probability from the same logits a clean signal would produce.
- **MC Dropout** captures variability in the model's own decision boundary, not in the inputs' signal quality.
- **Deep Ensembles** capture variability across trained members; again, a signal-property-agnostic quantity.

This is why LW-CCSD adds value (+4.1 %, +6.7 %, +15.6 %) on the first three sources and is redundant (−2.3 %) on EDL.

## Implications for the paper

1. **The cross-UQ table grows to 4 methods.** Section 5.4 + Table 4 now report all four: deterministic, MC Dropout, ensembles, EDL. The +15.6 %, +4.1 %, +6.7 %, −2.3 % progression is the cleanest evidence yet that LW-CCSD is a substitute for (not a complement to) sufficient UQ.

2. **The deployment punchline strengthens further.** Previously: deterministic + LW-CCSD beats clean ensemble UQ-only at 1/5 the compute. Now: EDL UQ-only (0.1725) beats *all* other deployments including deterministic + LW-CCSD (0.1998) by 13.7 % relative, also at 1/5 the compute. The recommendation chain becomes:

  - Smallest compute budget: deterministic + LW-CCSD (0.1998)
  - Standard budget: **single-pass EDL** (0.1725) — clear winner
  - Maximum budget: clean ensemble + LW-CCSD (0.1928)

  EDL at single-pass inference cost dominates ensembles in both directions.

3. **Section 5.12 (new) documents the mechanism quantitatively.** EDL evidence collapse on noisy inputs is the in-method realisation of what LW-CCSD does externally. This is a falsifiable mechanism prediction: any UQ method that conditions on input signal quality directly should similarly admit no LW-CCSD margin.

4. **Section 6.2 (deployment) is updated:** EDL is the recommended UQ method when a single-pass inference budget can accommodate the Dirichlet head training overhead (~one extra training run, no inference-time cost beyond softplus).

## Limitations of this EDL result

- **One checkpoint, no ensemble of EDL members.** Variance across `model_seed` not characterised. An EDL ensemble may further improve AURC but is not the topic of this paper.
- **KL annealing schedule fixed at 10 epochs.** Sensitivity not explored. Sensoy et al. report robustness to annealing length but the synth-rPPG substrate may differ.
- **EDL is poorly calibrated on this run (ECE 0.137).** A user who cares about reliability diagrams should still pick the clean ensemble. AURC and ECE rank UQ methods differently here.
- **No conformal extension.** The CP-LCB / Holm / Bonferroni conformal machinery added in Sections 5.6 transfers to EDL (the val recall counts are computed identically) but was not re-run here. Reported numbers are empirical-floor only.

## Files

- Model wrapper: `src/rppg_sa/models/edl_classifier.py`.
- Loss and uncertainty utilities: `src/rppg_sa/uncertainty/evidential.py` (already present).
- Trainer integration: `scripts/train_classifier.py` `--head evidential` flag with annealed KL.
- Eval integration: `scripts/eval_selective.py` `--uq evidential`.
- Val-dump integration: `scripts/dump_val_predictions.py` `--uq evidential`.
- Trained checkpoint: `runs/synth_rppg_cinc_edl/best.pt` (val macro-F1 = 0.6179 at epoch 25).
- Test eval: `runs/synth_rppg_cinc_edl/eval_evidential/`.
- Val predictions: `runs/synth_rppg_cinc_edl/eval_val_evidential/predictions.csv`.
- LW-CCSD floor sweep: `runs/synth_rppg_cinc_edl/eval_evidential/lw_ccsd_clean_edl_floor*/`.
- This findings document.

## Next

- Train a 3-member EDL ensemble for variance characterisation.
- Re-run with KL annealing 5, 15, 25 epochs to map calibration vs AURC trade.
- Replicate on the OBF / MAHNOB-HCI face-video AF data when access lands; the EDL mechanism prediction (no LW-CCSD margin) is the key falsification target.
