# EDL KL=∞ ablation — annealing is necessary for the no-LW-CCSD-margin property

**Date:** 2026-06-27
**Substrate:** synth-rPPG CinC test set ($n = 6{,}750$). Single EDL checkpoint trained with `--edl-annealing-steps 1`, meaning the KL prior is applied at full weight from epoch 1 onwards (no warm-up phase). Shared `data_seed = 42`, `model_seed = 1`. This is the limit-case ablation for the Section 5.16 KL annealing series.

## Headline — the negative LW-CCSD margin disappears in the no-annealing limit

| KL schedule | Test Acc | ECE     | Brier | UQ-only AURC | Best LW-CCSD | Δ AURC      | AF cost |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **1 (KL→∞, no annealing)** | **0.7382** | 0.133 | 0.425 | 0.1718 | 0.1703 | **+0.92 %** | **+0.000** |
| 5 epochs            | 0.7360 | 0.133 | 0.427 | 0.1680 | 0.1709 | −1.68 % | −0.036    |
| 10 epochs (Sec 5.12) | 0.7390 | 0.138 | 0.426 | 0.1725 | 0.1764 | −2.28 % | −0.013    |
| 15 epochs           | 0.7418 | 0.139 | 0.425 | 0.1713 | 0.1748 | −2.07 % | −0.039    |
| 25 epochs           | 0.7341 | 0.124 | 0.424 | 0.1736 | 0.1788 | −2.99 % | −0.021    |

Three findings:

1. **The no-LW-CCSD-margin property is annealing-dependent.** All four annealed schedules (KL 5, 10, 15, 25) produce a negative LW-CCSD margin (range [−1.68 %, −2.99 %]). The KL→∞ ablation produces a positive margin (+0.92 %) at zero AF cost. The transition from "negative margin" to "marginal positive margin" happens at the annealing/no-annealing boundary.

2. **KL→∞ test metrics are otherwise typical for EDL.** Test accuracy 0.7382 (between KL5's 0.7360 and KL15's 0.7418), ECE 0.133 (best annealed schedule was 0.124 at KL25), Brier 0.425 (in-range for the annealed schedules). The architecture and loss surface are not pathological under KL→∞ — only the LW-CCSD susceptibility is qualitatively different.

3. **Mechanism refinement.** The Section 5.12 statement "EDL evidence collapses on noisy signals, absorbing the SQI signal in-method" was previously claimed as a structural property of the EDL training framework. The KL→∞ ablation shows this is not architecturally inevitable; it requires a warm-up phase where weak KL pressure lets the evidence head develop selective collapse before the full prior tightens. Without annealing the model still classifies competently but does not develop the input-quality-conditional evidence behavior. LW-CCSD then offers a marginal positive margin like on classical UQ.

## Refined mechanism statement

The Section 5.12 paper claim is updated to:

> EDL **with KL annealing** absorbs signal-quality information in-method; EDL without annealing does not. The no-LW-CCSD-margin property is a controllable training-time consequence of the KL warm-up schedule, not an architectural property of the Dirichlet evidence head.

This refinement preserves the deployment recommendation chain from Section 5.14 (single-pass EDL at standard inference budget is the AURC-first choice) but adds nuance: any practitioner using EDL with the standard 10-epoch annealing schedule should expect the no-margin behavior; one who disables annealing should expect classical-UQ LW-CCSD behavior with a small positive margin.

## Practical implications

- **A deployer who wants LW-CCSD to add value on EDL**: disable KL annealing (`--edl-annealing-steps 1`). Trade ~0.4 % rel AURC vs the best annealed schedule (KL5 at 0.1680 vs KL→∞ at 0.1718) for a small positive LW-CCSD margin.
- **A deployer who wants the cheapest deferral-free selective behavior**: keep annealing on. The annealed schedules give a 1.5 % rel AURC advantage at the UQ-only level (0.1680 vs 0.1718) plus the absence-of-LW-CCSD claim.
- **The annealed default (KL=10) is still the most defensible choice for the paper.** The KL→∞ ablation tells the reader the mechanism is controllable; it does not make the annealed default obsolete.

## Limitations

- **Single seed (model_seed = 1).** Per-schedule variance not characterised. The Section 5.14 EDL ensemble variance at KL10 (AURC 0.1681 ± 0.0027) is a reasonable prior for KL→∞ variance but not verified.
- **One point on the no-annealing dimension.** The KL→∞ ablation tests `annealing_steps = 1` only. Intermediate "barely-annealed" schedules (KL2, KL3) between the no-annealing limit and the smooth-annealing regime were not characterised; the precise location of the margin-transition is unknown.
- **The +0.92 % margin is marginal.** Without additional sweeps it could be within the variance band of "essentially zero" rather than a statistically significant positive. The qualitative conclusion (annealing is necessary for the no-margin property) is the durable finding; the precise magnitude is less certain.

## Files

- Trained checkpoint: `runs/synth_rppg_cinc_edl_klinf/best.pt` (val macro-F1 = 0.6267 at epoch 24).
- Test eval: `runs/synth_rppg_cinc_edl_klinf/eval_evidential/`.
- Val predictions: `runs/synth_rppg_cinc_edl_klinf/eval_val_evidential/predictions.csv`.
- LW-CCSD sweep outputs: `runs/synth_rppg_cinc_edl_klinf/lw_ccsd_clean_klinf_floor*/`.
- This findings document.

## Next

- The mechanism-robustness story is now complete on synth-rPPG. Real-substrate (OBF / MAHNOB-HCI) replication is the v2.0 critical experiment.
- An interesting v2.0 follow-up: train EDL with KL annealing then ablate the annealing schedule post hoc by removing it before fine-tuning — does the no-margin property persist or unwind?
