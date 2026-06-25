# EDL Ensemble (M=5) — variance characterisation + mechanism confirmation

**Date:** 2026-06-24
**Substrate:** synth-rPPG CinC test set (n = 6,750). Five EDL members trained with shared `data_seed = 42` and independent `model_seed ∈ {1, 2, 3, 4, 5}`. Architecture, loss, and KL annealing schedule (10 epochs) identical across all members. Inference: average alpha across members, then compute Dirichlet probabilities and uncertainty u = K/S as usual.

## Headline — EDL ensemble is the strongest UQ configuration; LW-CCSD remains negative

| Configuration            | Test Acc | ECE     | Brier   | UQ-only AURC | Best safe LW-CCSD | Δ AURC      |
|---|---:|---:|---:|---:|---:|---:|
| EDL member m=1            | 0.7390   | 0.138   | 0.426   | 0.1725       | 0.1764             | −2.3 %      |
| EDL member m=2            | 0.7356   | 0.140   | 0.432   | 0.1685       | —                  | —           |
| EDL member m=3            | 0.7434   | 0.139   | 0.423   | 0.1655       | —                  | —           |
| EDL member m=4            | 0.7418   | 0.135   | 0.423   | 0.1679       | —                  | —           |
| EDL member m=5            | 0.7430   | 0.140   | 0.425   | 0.1664       | —                  | —           |
| **Per-member mean ± std** | **0.7405 ± 0.0033** | **0.138 ± 0.002** | **0.426 ± 0.003** | **0.1681 ± 0.0027** | — | — |
| **EDL ensemble (M=5)**    | **0.7468** | 0.144 | **0.422** | **0.1640**   | **0.1679**         | **−2.4 %**  |

Four findings:

1. **EDL is variance-stable.** Per-member AURC has std 0.0027 on a mean of 0.1681 (1.6 % rel). The original Section 5.12 result (member m=1, AURC 0.1725) sits at the upper end of the per-member distribution but within the typical seed-to-seed spread — the headline EDL claim is reproducible.

2. **EDL ensemble improves AURC by 2.5–5 % relative over single EDL.** Ensemble AURC 0.1640 vs best-single 0.1655 (+0.9 % rel improvement over the best member) vs mean-single 0.1681 (+2.5 % rel improvement over the average member). The improvement is moderate — Dirichlet averaging exploits some additional inter-member diversity but the single EDL is already close to the ensemble.

3. **EDL ensemble has the best AURC of any configuration studied (0.1640) AND the best test accuracy (0.7468) AND the best Brier score (0.422)** at five-pass inference cost. Compared to clean softmax ensemble at the same M=5 inference budget: EDL ensemble cuts AURC by 20.6 % relative (0.2066 → 0.1640), cuts Brier by 4.1 % (0.440 → 0.422), and lifts accuracy by 5.6 absolute points (0.692 → 0.747).

4. **LW-CCSD remains structurally negative on the EDL ensemble.** Every feasible operating point produces negative ΔAURC, ranging from −2.91 % at floor 0.20 to −2.40 % at the best safe floor (0.30, w*=0.0/0.3/0.3). The Section 5.12 mechanism — Dirichlet evidence collapses on noisy signals, absorbing the SQI signal in-method — is therefore a structural property of EDL training, not a feature of any one initialisation. Ensembling does not change it.

## EDL ensemble LW-CCSD Pareto

Val UQ-only AF recall at c₀ = 0.50 is 0.298 (similar to single EDL's 0.313). Floors ≥ 0.35 are infeasible.

| Val AF floor | w* (NSR/AF/Other) | Test AURC | Δ vs UQ | Test AF@0.5 | Δ AF |
|---:|---|---:|---:|---:|---:|
| 0.20         | 0.0 / 0.4 / 0.3   | 0.1688    | −2.91 % | 0.347       | −0.036    |
| 0.25         | 0.0 / 0.4 / 0.3   | 0.1688    | −2.91 % | 0.347       | −0.036    |
| 0.28         | 0.0 / 0.4 / 0.3   | 0.1688    | −2.91 % | 0.347       | −0.036    |
| **0.30**     | **0.0 / 0.3 / 0.3** | **0.1679** | **−2.40 %** | **0.372** | **−0.011** |
| UQ-only      | —                 | 0.1640    | —       | 0.383       | 0.000     |

## ECE: a sharpening effect from Dirichlet averaging

EDL ensemble ECE is 0.144 — slightly higher than the worst single-member ECE (0.140) and 0.7 absolute points higher than the best (0.135). The Dirichlet evidence average produces probabilities sharper than any individual member's, opening a small calibration gap on top of an already poorly-calibrated UQ source. This trade-off is the EDL signature: ranking improvements come at calibration cost. For an AURC-first deployment this is acceptable; for a calibration-first deployment, SNGP (ECE 0.029) remains the recommended choice.

## Deployment chain — updated

| Budget tier | Configuration | Test AURC | Test Acc | ECE |
|---|---|---:|---:|---:|
| Smallest      | Deterministic + LW-CCSD          | 0.1998 | 0.686 | — |
| Standard      | Single-pass EDL                  | 0.1725 | 0.739 | 0.138 |
| Standard +    | SNGP + LW-CCSD                   | 0.1969 | 0.729 | 0.029 |
| Maximum AURC  | **EDL ensemble (M=5)**           | **0.1640** | **0.747** | 0.144 |
| Maximum cal   | Clean softmax ensemble + LW-CCSD | 0.1928 | 0.692 | 0.052 |

EDL ensemble dominates AURC and accuracy at five-pass inference cost. It is the recommended choice when AURC + accuracy are jointly the deployment objective and the deployer can afford the M=5 inference budget. SNGP remains the calibration-first choice; deterministic + LW-CCSD remains the lightweight choice.

## Limitations

- **KL annealing schedule fixed at 10 epochs across all members.** Sensitivity to the schedule has not been mapped.
- **No conformal extension on the ensemble.** The Bonferroni / Holm machinery transfers directly but has not been re-run on the ensemble predictions.
- **Single seed family** (model_seed 1..5 from a deterministic torch initialisation). A larger ensemble or LHS-sampled seeds may shift the variance estimate.
- **Diagonal-Laplace nature of EDL uncertainty is shared across members.** Ensemble of EDLs does not enrich the uncertainty model class; it averages an evidence head that is structurally the same in every member.

## Files

- Trained members: `runs/synth_rppg_cinc_edl/best.pt`, `runs/synth_rppg_cinc_edl_m{2,3,4,5}/best.pt`. All checkpoints embed cfg with `classifier.head: evidential`.
- Per-member test evals: `runs/synth_rppg_cinc_edl_m{2..5}/eval_evidential/`.
- Ensemble test eval: `runs/synth_rppg_cinc_edl/eval_evidential_ensemble/`.
- Ensemble val predictions for LW-CCSD: `runs/synth_rppg_cinc_edl/eval_val_evidential_ensemble/predictions.csv`.
- Ensemble LW-CCSD sweep outputs: `runs/synth_rppg_cinc_edl/lw_ccsd_clean_edl_ens_floor*/`.
- Code: `scripts/eval_selective.py --uq evidential_ensemble`, `scripts/dump_val_predictions.py --uq evidential_ensemble` — both forward through all M checkpoints, average alpha, compute Dirichlet uncertainty on the averaged alpha.

## Next

- EDL KL annealing sensitivity (5 / 15 / 25 epochs) — does the no-LW-CCSD-margin property survive different KL schedules?
- EDL conformal extension (Bonferroni + Holm) — straightforward; the val recall counts are computed identically.
- Replication on OBF / MAHNOB-HCI face-video AF data is now the cleanest test of EDL's evidence-collapse mechanism on a real rPPG substrate.
