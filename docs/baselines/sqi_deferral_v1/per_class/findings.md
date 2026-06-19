# Per-class SNR-deferral analysis — the 18 % AURC gain is class-asymmetric

**Date:** 2026-06-18
**Substrate:** CinC test set (n=6,750), deterministic checkpoint `runs/synth_rppg_cinc/best.pt`.
**Comparison:** UQ-only deferral (w=0) vs SNR-weighted deferral (w=0.70), the configuration that gave the +18 % aggregate AURC gain.

## Headline reversal

The aggregate AURC win comes **at the cost of catastrophic AF rejection**. The mechanism is intrinsic: AF is defined by irregular rhythm, which produces low spectral SNR in the cardiac band, so SNR-deferral systematically rejects the very signature the model uses to detect AF.

| Coverage | AF recall UQ | AF recall SQI w=0.70 | Δ AF | NSR recall UQ | NSR recall SQI | Δ NSR |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | **0.707** | **0.043** | **−0.664** | 0.416 | 0.584 | +0.168 |
| 0.70 | 0.732 | 0.365 | −0.367 | 0.641 | 0.741 | +0.100 |
| 0.80 | 0.764 | 0.637 | −0.127 | 0.730 | 0.780 | +0.050 |
| 0.90 | 0.800 | 0.730 | −0.070 | 0.797 | 0.819 | +0.022 |
| 0.95 | 0.814 | 0.778 | −0.036 | 0.820 | 0.836 | +0.016 |
| 1.00 | 0.828 | 0.828 |   0.000 | 0.841 | 0.841 |  0.000 |

At 50 % coverage, SNR-deferral keeps only **54 of 559 AF segments** (vs UQ's 431). The model finds AF perfectly well on those AF segments (precision stays at ~0.92), but the deferral rule throws nearly all of them away before classification.

## n-kept counts at 50 % coverage

| Class | n_total | n_kept UQ | n_kept SQI w=0.70 |
|---|---:|---:|---:|
| NSR | 4,077 | 1,893 | **2,466** |
| AF | 559 | 431 | **54** |
| Other | 2,114 | 1,051 | 855 |

SQI deferral redistributes coverage toward NSR. AF and Other are systematically rejected.

## Why this happens

Spectral SNR is computed as the ratio of in-band power (0.7–3 Hz, the cardiac band) to out-of-band noise. Regular rhythm → a sharp spectral peak at the heart rate → high SNR. Irregular rhythm (AF, ectopy) → distributed power across the cardiac band → broader peak → lower SNR.

The deferral score combines model confidence with SNR rank-normalized. Even though the model is moderately confident on AF (mean entropy ~0.43 for ensembles), the SNR rank for AF segments is systematically low because their rhythm is irregular *by definition*. With w=0.70 the SNR term dominates and AF gets dumped.

**The same feature the model uses to detect AF is what the deferral rule penalizes.** This is intrinsic, not a bug in the SNR estimator.

## Reframing the paper

Original headline: "SNR-weighted deferral beats UQ-only deferral by 18 % AURC."

Corrected headline: "Naive SNR-weighted deferral improves aggregate AURC by 18 %, **but the gain is clinically inverted**: AF recall collapses from 0.71 to 0.04 at 50 % coverage. The mechanism is intrinsic — AF's irregularity *is* a low-SNR signature — so safe deferral must be class-conditional."

This is the stronger paper. The negative result is the contribution.

## Class-conditional deferral — the fix

Instead of a single deferral score, weight signal quality only for the class where it's appropriate:

  combined(x) = (1 − w) · model_confidence(x) + w · SNR(x)        if predicted class ∈ {NSR, Other}
              = model_confidence(x)                                if predicted class = AF

In words: keep AF predictions as long as the model is confident, regardless of SNR. Defer NSR predictions when SNR is low. Same SNR feature, applied where it carries the right signal.

Predicted variants of this rule worth comparing:

1. **AF-immune SQI deferral** as above (binary: weight SNR for non-AF predictions only).
2. **Per-class quality weight**: w_NSR, w_AF, w_Other learned to optimize per-class recall floor.
3. **Calibrated class-conditional**: SNR is rank-normalized within each predicted class before combination.

Test all three on the same CinC checkpoint. Headline metric: AURC at fixed AF-recall floor (e.g. AF recall ≥ 0.70).

## Files

- Code: `scripts/eval_sqi_deferral_per_class.py`.
- Per-class table: `runs/synth_rppg_cinc/eval_conformal/per_class_sqi_deferral/per_class_snr_db_w0.70.csv`.

## Next

1. **Implement class-conditional deferral** as above; rerun per-class breakdown. Headline metric: AURC subject to AF recall ≥ 0.70 floor.
2. **Replicate per-class on MC Dropout + ensembles.** Verify the AF-collapse mechanism is consistent across UQ methods.
3. **Per-class on MIT-BIH classifier** as the cross-dataset robustness check.
4. **Update the paper outline** — the contribution is now a negative result + a class-conditional fix, not the naive SQI claim.
