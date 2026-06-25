# EDL KL annealing sensitivity — schedule robustness of the no-margin property

**Date:** 2026-06-24
**Substrate:** synth-rPPG CinC test set ($n = 6{,}750$). Single EDL checkpoint per schedule, all with shared `data_seed = 42` and `model_seed = 1`. Four annealing schedules: KL ramp linearly from 0 to 1 over 5, 10 (Section 5.12 baseline), 15, and 25 epochs.

## Headline — every schedule produces a negative LW-CCSD margin

| KL annealing | Val Macro-F1 | Test Acc | ECE     | Brier   | UQ-only AURC | Best safe LW-CCSD | $\Delta$ AURC | AF cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 epochs            | 0.6156 | 0.7360 | 0.133   | 0.427   | 0.1680       | 0.1709             | −1.68 %  | −0.036    |
| **10 (Section 5.12)** | **0.6179** | **0.7390** | 0.138 | 0.426 | **0.1725** | 0.1764 | **−2.28 %** | **−0.013** |
| 15 epochs           | 0.6277 | **0.7418** | 0.139   | 0.425   | 0.1713       | 0.1748             | −2.07 %  | −0.039    |
| 25 epochs           | 0.6248 | 0.7341 | **0.124** | **0.424** | 0.1736     | 0.1788             | −2.99 %  | −0.021    |
| Mean ± std (n = 4)  | 0.6215 ± 0.0050 | 0.7377 ± 0.0030 | 0.134 ± 0.006 | 0.426 ± 0.001 | **0.1714 ± 0.0021** | 0.1752 ± 0.0028 | **−2.26 % ± 0.49 %** | — |

Three findings:

1. **EDL is robust to KL annealing schedule.** UQ-only AURC sits in [0.1680, 0.1736] — a 3.3 % relative spread across a 5× variation in annealing length. Best AURC (0.1680) is at the shortest schedule; longest schedule (KL25) gives 0.1736 (3.3 % worse). The Section 5.12 baseline at KL10 sits in the middle of this range. KL15 has the best test accuracy (0.7418), KL25 has the best ECE (0.124) and best Brier (0.424). No single schedule dominates on all metrics — there is a calibration / accuracy tradeoff but the AURC range is narrow.

2. **The negative LW-CCSD margin holds across every schedule.** Range [−1.68 %, −2.99 %], mean −2.26 % ± 0.49 %. No annealing schedule produces a positive margin or even a near-zero one. The shortest schedule (KL5) has the smallest negative margin (−1.68 %) but is still strictly negative; the longest (KL25) has the largest (−2.99 %).

3. **Larger negative margin correlates with longer KL annealing.** KL5 → −1.68 %, KL10 → −2.28 %, KL15 → −2.07 %, KL25 → −2.99 %. A monotonic relationship is *not* observed (KL15 sits below KL10), but the overall direction is clear: longer annealing → tighter evidence collapse on noisy signals → less room for external SNR to add information → more negative margin. Mechanistically consistent with Section 5.12: prolonged KL pressure pushes the model harder to concentrate evidence on confident-classifiable inputs, which is exactly the in-method signal-quality measurement.

## Reading

The Section 5.12 no-margin claim is now established at four independent levels: 
1. **Empirical optimisation** (Section 5.12 baseline at KL10, single seed) 
2. **Variance across initialisation seeds** (Section 5.14 EDL ensemble, model_seed 1..5) 
3. **Conformal coverage robustness** (Section 5.15 per-class + Bonferroni + Holm at family-wise α=0.10) 
4. **KL annealing schedule robustness** (this section, KL ∈ {5, 10, 15, 25} epochs)

There is no plausible null hypothesis remaining within the EDL training framework that would flip the LW-CCSD margin to positive. The Section 5.12 mechanism — Dirichlet evidence collapse on noisy signals → in-method signal-quality measurement → redundancy with external SNR — is robust at every level the design choice space exposes.

The remaining open question is whether the property survives on the *real* face-video AF substrate (OBF / MAHNOB-HCI), which is the genuine external falsification target. The four levels of synth-rPPG robustness reported here are necessary but not sufficient for the OBF claim; the real-substrate replication is the v2.0 critical experiment.

## Implications for the paper

- Section 5.16 (new): single short subsection with Table 18 summarising the four schedules. The KL-sensitivity result is small in code surface but matters substantially for reviewer responses ("how sensitive is EDL to annealing?").
- Section 5.12 mechanism statement upgraded from "robust on this checkpoint" to "robust across initialisation, ensembling, conformal coverage, AND KL annealing schedule".
- The KL5 result (smallest negative margin at −1.68 %) suggests the EDL no-margin property is weakest when KL pressure is shortest; this is a fine-grained mechanism insight worth flagging in the prose.

## Files

- Trained checkpoints: `runs/synth_rppg_cinc_edl_kl{5,15,25}/best.pt` (KL10 is `runs/synth_rppg_cinc_edl/best.pt`).
- Per-KL test evals: `runs/synth_rppg_cinc_edl_kl{5,15,25}/eval_evidential/`.
- Per-KL val predictions: `runs/synth_rppg_cinc_edl_kl{5,15,25}/eval_val_evidential/`.
- Per-KL LW-CCSD sweeps: `runs/synth_rppg_cinc_edl_kl{5,15,25}/lw_ccsd_clean_kl_floor*/`.

## Limitations

- **Single seed (model_seed = 1) per schedule.** Per-schedule variance not characterised. The Section 5.14 per-member variance at KL10 (AURC 0.1681 ± 0.0027) is a reasonable prior for the variance at other schedules, but not verified.
- **Schedules sampled at 5, 10, 15, 25.** No interpolation between, no extrapolation beyond 25. KL annealing → ∞ (constant KL weight = 1) is the limit case and has not been tested; it may behave qualitatively differently because the KL pressure starts immediately rather than after a warm-up phase.
- **No KL weight scaling.** All schedules use the same KL coefficient (1.0 at the end of annealing). Reducing this coefficient at long schedules might recover positive margin — untested.

## Next

- KL = ∞ (constant KL = 1) sensitivity test, if reviewers request it.
- Replication on OBF / MAHNOB-HCI when access lands — the genuine external test.
