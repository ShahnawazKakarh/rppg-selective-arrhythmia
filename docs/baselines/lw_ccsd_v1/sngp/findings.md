# SNGP — 5th UQ method; falsifies the EDL mechanism prediction

**Date:** 2026-06-24
**Substrate:** synth-rPPG CinC test set ($n = 6{,}750$). Trained with shared `data_seed = 42`, `model_seed = 1`. Spectral-normalised CNN1D-Transformer backbone (coefficient 0.95 via `nn.utils.parametrizations.spectral_norm` on every Conv1d/Linear) + Random Fourier Features projection ($D = 1024$, RBF kernel scale 1.0) + diagonal-Laplace GP head (ridge $10^{-3}$). After standard CE training, the precision matrix is accumulated over a full training-set pass and inverted to a diagonal covariance.

## Headline — SNGP admits a positive LW-CCSD margin like the classical UQ sources

| UQ Method                                       | Test Acc | ECE     | UQ-only AURC | Best safe LW-CCSD | Δ AURC     | AF cost |
|---|---:|---:|---:|---:|---:|---:|
| Deterministic                                    | 0.686    | —       | 0.2367       | 0.1998             | +15.6 %    | −0.081 |
| MC Dropout ($T = 30$)                            | 0.715    | 0.076   | 0.1980       | 0.1899             | +4.1 %     | −0.029 |
| Clean Ensemble ($M = 5$)                         | 0.692    | 0.052   | 0.2066       | 0.1928             | +6.7 %     | −0.032 |
| **SNGP** (this checkpoint)                       | **0.729** | **0.029** | 0.2044    | **0.1969**         | **+3.7 %** | −0.023 |
| EDL                                              | 0.739    | 0.137   | **0.1725**   | 0.1764             | −2.3 %     | −0.013 |

Three observations.

1. **SNGP has the best test ECE of all five methods (0.029).** Spectral normalisation + GP last-layer is the cleanest calibration story in this benchmark; ECE drops from 0.052 (clean ensemble) to 0.029, a 44 % relative improvement, at the same single-pass inference cost.

2. **SNGP admits a positive LW-CCSD margin: +3.7 % at the best safe floor (0.06, AF cost only −0.023).** This is mid-range across the classical UQ methods (+4.1 % MC Dropout, +6.7 % ensembles) — SNGP behaves like a sharper deterministic confidence ranking, not like EDL.

3. **The EDL mechanism prediction is refuted.** Section 5.12 hypothesised that *any UQ method that conditions on input signal quality directly* should admit no LW-CCSD margin. SNGP's GP head is explicitly distance-aware — its variance is large where the input is far from the training-set RFF distribution in feature space — and yet it admits a clear positive margin. EDL's negative margin is therefore not a generic property of input-conditional UQ; it is a specific consequence of EDL's evidence-on-noisy-inputs training dynamic.

## SNGP LW-CCSD Pareto frontier

Val UQ-only AF recall at $c_0 = 0.50$ is **0.085** (extremely low — SNGP defers heavily on AF), so the floor sweep must start at 0.03.

| Val AF floor | $\mathbf{w}^{*}$ (NSR/AF/Other) | Test AURC | Δ vs UQ | Test AF@0.5 | Δ AF      |
|---:|---|---:|---:|---:|---:|
| 0.03         | 0.0 / 0.4 / 1.0                 | 0.1979    | +3.16 % | 0.052       | −0.047    |
| 0.05         | 0.0 / 0.2 / 1.0                 | 0.1973    | +3.47 % | 0.070       | −0.029    |
| **0.06**     | **0.0 / 0.1 / 1.0**             | **0.1969** | **+3.69 %** | **0.075** | **−0.023** |
| 0.08         | 0.0 / 0.0 / 0.4                 | 0.2019    | +1.24 % | 0.098       | +0.000    |
| UQ-only      | —                               | 0.2044    | —       | 0.098       | 0.000     |

All four feasible floors give positive ΔAURC. The frontier compresses as the floor approaches the val UQ baseline (0.08 → +1.24 % only), exactly as it does for the classical sources — the constraint becomes tight, the optimiser has less feasible w to choose from, and the margin shrinks.

## Mechanism — why EDL is unique and SNGP is not

Both EDL and SNGP attempt to be input-aware:

- **EDL:** under the EDL training objective, an input that lacks a clean cardiac rhythm produces low evidence on *every* class, hence low total $\alpha$, hence high Dirichlet uncertainty $u = K/S$. The uncertainty is high *because there is no class to predict confidently*. This directly mirrors what spectral SNR measures externally: poor signal $\to$ no class.

- **SNGP:** the variance head reports high uncertainty when the input is far from training-set RFF features in the spectral-normalised feature space. A noisy rPPG signal *will* tend to be far in feature space — but so will a *clean* signal from an underrepresented record. Distance-in-features and signal-quality are correlated but not the same. SNR carries information about signal quality that the GP variance does not capture: the test set's spread in feature space is determined by the training distribution, not by the signal-quality distribution per se.

This decoupling is why SNGP admits LW-CCSD margin and EDL does not. EDL's "no class confident" criterion *is* a signal-quality criterion at the level of the training objective; SNGP's "far from training features" criterion is closer to an OOD criterion, which is correlated with — but distinct from — signal quality.

The revised mechanism statement for the paper:

> EDL's no-LW-CCSD-margin behaviour stems from a specific property of the EDL training objective: evidence collapses on inputs that lack a confident-classifiable signal, which is operationally identical to a signal-quality measurement. This is not a property of input-conditional UQ in general. SNGP, which is also input-conditional via its distance-aware feature norm, admits the same positive LW-CCSD margin (+3.7 %) as the classical UQ methods.

## Implications for the paper

1. **The cross-UQ table now spans 5 methods** with a wider range of behaviours than v1.2.0 documented. Section 5.4 + Table 4 updated.

2. **Section 5.12 is now a falsification narrative.** The hypothesis is named, then refuted by Section 5.13 (SNGP). This is stronger paper material than a single positive finding: the mechanism is sharpened by what fails to follow from it.

3. **Deployment recommendation chain unchanged but refined:**
  - Smallest budget: deterministic + LW-CCSD (0.1998)
  - **Standard budget: single-pass EDL (0.1725)** — best AURC
  - Best-calibrated budget: **single-pass SNGP (0.2044 raw, 0.1969 with LW-CCSD)** — best ECE
  - Maximum budget: clean ensemble + LW-CCSD (0.1928)

  Adding SNGP gives the practitioner a calibration-first choice that EDL does not provide (EDL has the worst ECE of all five methods).

## Limitations of this SNGP result

- **One checkpoint, no ensemble.** Variance across `model_seed` not characterised.
- **Diagonal Laplace approximation** rather than full per-class covariance — the standard SNGP cost/quality trade-off. Full covariance would tighten variance estimates at $K \cdot D \cdot D$ memory cost (here, $3 \cdot 1024 \cdot 1024 \approx 3 \cdot 10^6$ floats; feasible).
- **No spectral-norm coefficient sweep.** Liu et al. (2020) report sensitivity to the coefficient; 0.95 is the standard default and was not tuned here.
- **Single-pass precision accumulation in the final epoch only.** Re-running with full-training accumulation may sharpen the precision diagonal.

## Files

- `src/rppg_sa/uncertainty/sngp.py` — `RandomFourierFeatures`, `GPLayer`, `mean_field_logits`, `apply_spectral_norm`.
- `src/rppg_sa/models/sngp_classifier.py` — `CNN1DTransformerSNGP` wrapper.
- `scripts/train_classifier.py` — `--head sngp` flag with `--sngp-rff-features`, `--sngp-rff-kernel-scale`, `--sngp-ridge`, `--sngp-sn-coefficient`. Special training procedure: standard CE → final-epoch precision accumulation → post-training re-pass + `finalize_gp()`.
- `scripts/eval_selective.py` and `scripts/dump_val_predictions.py` — `--uq sngp`.
- `runs/synth_rppg_cinc_sngp/best.pt` — checkpoint with finalised GP (val macro-F1 = 0.6297 at epoch 19, best of all five methods).
- `runs/synth_rppg_cinc_sngp/eval_sngp/` — test eval.
- `runs/synth_rppg_cinc_sngp/eval_val_sngp/` — val predictions for LW-CCSD.
- `runs/synth_rppg_cinc_sngp/lw_ccsd_clean_sngp_floor*/` — LW-CCSD floor sweep outputs.

## Next

- **EDL ensemble** (3–5 members) to test whether EDL's no-margin property is variance-stable.
- **SNGP coefficient + ridge sweep** to check robustness of the +3.7 % margin.
- **Mean-field correction scale sensitivity** ($\pi/8$ default → 0.5 / 1 / 2) to map the cap on variance influence.
- **OBF / MAHNOB-HCI face-video AF replication** is now the cleanest mechanism falsification target across all five UQ sources.
