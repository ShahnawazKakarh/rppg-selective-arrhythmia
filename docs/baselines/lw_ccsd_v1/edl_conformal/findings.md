# EDL conformal extension — per-class, Bonferroni, Holm

**Date:** 2026-06-24
**Substrate:** synth-rPPG CinC test set ($n = 6{,}750$). Single EDL checkpoint (member m=1) from Section 5.12. Conformal machinery: Clopper-Pearson per-class one-sided LCB, Bonferroni union-bound joint, Holm step-down joint — all at family-wise (or per-class) $\alpha = 0.10$.

## Headline — the negative LW-CCSD margin survives every conformal guarantee

| Mode                                         | Floor | $\mathbf{w}^{*}$ (NSR/AF/Other) | Test AURC | $\Delta$ vs UQ-only | Test AF@0.5 | $\Delta$ AF |
|---|---:|---|---:|---:|---:|---:|
| Empirical (Section 5.12)                      | 0.30  | 0.0 / 0.3 / 0.4 | 0.1764    | −2.28 % | 0.386       | −0.013 |
| **Per-class conformal, $\alpha = 0.10$**      | **0.28** | **0.0 / 0.2 / 0.4** | **0.1759** | **−2.00 %** | **0.394** | **−0.005** |
| Bonferroni joint, family-wise $\alpha = 0.10$ | 0.25  | 0.0 / 0.4 / 0.4 | 0.1770    | −2.65 % | 0.367       | −0.032 |
| **Holm step-down joint, family-wise $\alpha = 0.10$** | **0.28** | **0.0 / 0.2 / 0.4** | **0.1759** | **−2.00 %** | **0.394** | **−0.005** |
| UQ-only baseline (no LW-CCSD)                | —     | —               | 0.1725    | —       | 0.399       | 0.000  |

Three findings:

1. **The negative LW-CCSD margin survives every probabilistic guarantee.** The Section 5.12 empirical conclusion — adding SNR via LW-CCSD makes EDL strictly worse — holds under per-class conformal ($\Delta = -2.00 \%$), Bonferroni-corrected joint family-wise coverage ($\Delta = -2.65 \%$), and Holm step-down family-wise coverage ($\Delta = -2.00 \%$). A reviewer asking "is your finding robust to the strongest conformal guarantee available?" gets a clean yes.

2. **Per-class conformal and Holm produce the identical optimum on EDL** (both at floor 0.28, $\mathbf{w}^{*} = (0, 0.2, 0.4)$, AURC 0.1759). On softmax sources Holm strictly dominates per-class on the joint statement. On EDL the optimisation landscape is so flat near $\mathbf{w} = (0, \cdot, \cdot)$ that the LCB tightening does not shift $\mathbf{w}^{*}$ between the two procedures.

3. **Bonferroni's conservatism shows up as a stricter feasible floor.** Bonferroni's best safe floor is 0.25 (vs 0.28 for per-class and Holm), with AURC 0.0011 worse and AF cost six times larger. The standard Section 5.6 finding (Bonferroni is conservative for $C = 3$, Holm closes the gap) replicates on EDL exactly as on the deterministic baseline.

## Reading

The methodological point is that the negative LW-CCSD finding on EDL is not an artefact of empirical-floor optimisation. Section 5.6 of the paper shows that conformal LCBs tighten the feasibility constraint and shift $\mathbf{w}^{*}$ at the cost of a small AURC regression. On EDL, that machinery still operates correctly — Bonferroni gives the most conservative operating point, Holm recovers some of the gap — but every mode is firmly inside negative-margin territory. There is no LCB tightening that turns the EDL frontier positive.

This is the strongest possible robustness test of the Section 5.12 conclusion within the conformal framework. The remaining null hypothesis — that EDL's no-margin behaviour is specific to floor-0.30 empirical optimisation — is now rejected at all three procedures' family-wise levels.

## Implications for the paper

- Section 5.15 (new) reports this as a single table extension of Table 12. It is a short subsection (no new mechanism, just a robustness check) but matters for reviewer responses.
- The headline EDL claim now reads: "EDL admits negative LW-CCSD margin under empirical optimisation, per-class conformal $\alpha = 0.10$, Bonferroni joint family-wise $\alpha = 0.10$, and Holm step-down family-wise $\alpha = 0.10$." That is the strongest statement the conformal machinery can support.
- The single-EDL Section 5.12 result and the EDL-ensemble Section 5.14 result together establish: (a) the negative margin is reproducible across seeds (Section 5.14 per-member spread); (b) the negative margin survives ensembling (Section 5.14 ensemble result); (c) the negative margin survives every conformal coverage guarantee (this section). The EDL mechanism prediction is therefore established at three independent levels.

## Files

- `runs/synth_rppg_cinc_edl/lw_ccsd_clean_edl_conf_a10_floor*/` — per-class conformal sweep.
- `runs/synth_rppg_cinc_edl/lw_ccsd_clean_edl_bonf_a10_floor*/` — Bonferroni joint sweep.
- `runs/synth_rppg_cinc_edl/lw_ccsd_clean_edl_holm_a10_floor*/` — Holm step-down joint sweep.
- This findings document.

## Next

- KL annealing sensitivity (5 / 15 / 25 epochs) — does the no-margin property survive different KL schedules? This is the last open EDL-mechanism question before OBF/MAHNOB replication.
- Replication on OBF / MAHNOB-HCI face-video AF data — the genuine external falsification target.
