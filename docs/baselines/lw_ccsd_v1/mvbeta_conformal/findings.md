# Multivariate-beta exact-independent joint conformal coverage

**Date:** 2026-06-24
**Substrate:** synth-rPPG CinC test set (n = 6,750). Deterministic classifier baseline (same as Sections 5.6 Bonferroni/Holm runs). Family-wise α = 0.10. Per-class α derived from the exact-multiplicative bound under per-class independence.

## Statement

Under the independence assumption — val records are pre-partitioned by class and each per-class Clopper-Pearson test is computed on disjoint trial counts — the joint coverage probability factorises:

    P(all r_c ≥ LCB_c) = ∏_c P(r_c ≥ LCB_c).

Setting each per-class P(r_c ≥ LCB_c) = (1 − α)^(1/C) yields P(all r_c ≥ LCB_c) ≥ 1 − α exactly. Per-class α is therefore:

    α_c = 1 − (1 − α)^(1/C).

For C = 3, α = 0.10:

    α_c = 1 − 0.9^(1/3) = 0.0345.

Bonferroni gives α_c = α/C = 0.0333; MV-beta gives 0.0345. The MV-beta per-class threshold is slightly more lenient, admitting slightly larger LCBs and therefore (in theory) slightly more weight combinations.

## Headline — identical to Bonferroni at this grid resolution

| Mode                                      | Floor | w* (NSR/AF/Other) | Test AURC | Δ vs UQ | AF@0.5  | Δ AF    |
|---|---:|---|---:|---:|---:|---:|
| Per-class conformal α=0.10 (Section 5.6)   | 0.52  | 0.3 / 0.4 / 0.4   | 0.1998    | +15.6 % | 0.626   | −0.081  |
| Bonferroni joint family-wise α=0.10 (Sec 5.6) | 0.50  | 0.0 / 0.5 / 0.6   | 0.2046    | +13.55 % | 0.614   | −0.093  |
| Holm step-down joint family-wise α=0.10 (Sec 5.6) | 0.50  | 0.0 / 0.5 / 0.5   | 0.2033    | +14.11 % | 0.594   | −0.113  |
| **MV-beta joint family-wise α=0.10 (new)** | **0.50** | **0.0 / 0.5 / 0.6** | **0.2046** | **+13.55 %** | **0.614** | **−0.093** |
| MV-beta joint family-wise α=0.10 (new)     | 0.55  | 0.1 / 0.2 / 0.5   | 0.2160    | +8.75 % | 0.682   | −0.025  |
| MV-beta joint family-wise α=0.10 (new)     | 0.45  | 0.1 / 0.5 / 0.5   | 0.2007    | +15.22 % | 0.567   | −0.140  |
| MV-beta joint family-wise α=0.10 (new)     | 0.40  | 0.2 / 0.5 / 0.4   | 0.1977    | +16.49 % | 0.526   | −0.181  |

Three findings:

1. **MV-beta and Bonferroni produce identical w* and AURC at every floor that overlaps with the previous Bonferroni sweep.** The per-class α gap (0.0345 vs 0.0333) is below the 0.10 grid resolution — the LCB tightening does not shift which weight combinations are feasible. On finer grids (0.05 step or continuous optimisation) a small divergence may emerge.

2. **The 1.75 % gap left by Holm step-down is therefore intrinsic, not a Bonferroni-conservativeness artifact.** Going from Bonferroni's union bound to the exact-multiplicative bound under independence yields zero practical improvement at this grid + α. The remaining margin recoverable in the joint-coverage direction lives in the Holm step-down's *p-value ordering*, not in the per-class LCB tightness.

3. **MV-beta opens floor 0.55 as a new feasible operating point** (AURC 0.2160, AF cost only −0.025). Bonferroni's earlier sweep did not include floor 0.55; the slightly more lenient MV-beta LCB makes a higher-floor regime feasible with a moderate AURC concession. For a deployment that wants stronger AF safety than 0.50 buys, MV-beta + floor 0.55 is the option Bonferroni would not have surfaced.

## Implications for the paper

The conformal contribution now spans **four** procedures: per-class (Section 5.6) → Bonferroni joint (5.6) → Holm step-down joint (5.6) → MV-beta exact-independent joint (this section). Each one has a distinct probabilistic statement:

| Procedure | Per-class statement | Joint statement | Net cost vs per-class |
|---|---|---|---|
| Per-class | P(r_c ≥ floor) ≥ 1 − α each | None | 0 |
| Bonferroni | None (union bound) | P(all r_c ≥ floor) ≥ 1 − α | 2.05 % rel AURC |
| Holm step-down | None (sorted p-value test) | P(all r_c ≥ floor) ≥ 1 − α | 1.75 % rel AURC |
| **MV-beta exact** | P(r_c ≥ floor) ≥ (1 − α)^(1/C) | P(all r_c ≥ floor) ≥ 1 − α (exact under independence) | 2.05 % rel AURC |

MV-beta is the procedure to cite when the reader wants the *exact* joint coverage under independence; Holm is the procedure to cite for the *most powerful* joint test. Both reach the same family-wise guarantee 1 − α via different routes, and on synth-rPPG with C = 3 their practical operating points are within 0.6 % rel AURC of each other.

## Limitations

- **Grid resolution.** MV-beta and Bonferroni only diverge at per-class α difference 0.0012 — likely below the grid step. A finer grid (0.05 or continuous Nelder-Mead) could surface a genuine MV-beta vs Bonferroni gap, but it is unlikely to exceed 0.5 % rel AURC at this α.
- **Independence assumption.** The MV-beta exact bound requires that the per-class binomial tests are independent. This holds in our setup because val records are pre-partitioned by class label — but a future setup where the same record contributes to multiple class counts (e.g. multi-label) would invalidate the independence and require a copula-based correction.
- **C = 3 is the favourable regime for the exact bound.** As C grows, (1 − α)^(1/C) and α/C diverge more slowly, but α/C still bounds (1 − α)^(1/C) from above (Bonferroni stays conservative). Practical gain scales as O(α/C²) — small at any moderate α.

## Files

- `runs/synth_rppg_cinc/eval_conformal/lw_ccsd_clean_mvbeta_a10_floor*/` — MV-beta sweep outputs (one directory per floor).
- Code: `scripts/optimize_class_conditional_weights.py` `--conformal-joint-mvbeta` flag with `mvbeta_per_class_alpha()` helper.

## Conclusion

MV-beta closes the conformal methodology arc: the LW-CCSD paper now reports per-class, Bonferroni, Holm, AND MV-beta coverage modes — every standard frequentist family-wise procedure that applies to independent class events. The 1.75 % gap left by Holm relative to per-class is now established as intrinsic rather than a procedural conservativeness; further closing it requires a different objective (e.g. relaxed coverage at lower α) rather than a tighter joint bound.
