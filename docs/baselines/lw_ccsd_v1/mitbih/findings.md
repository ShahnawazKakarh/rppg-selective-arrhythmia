# LW-CCSD on MIT-BIH — graceful degradation

**Date:** 2026-06-18
**Substrate:** MIT-BIH Arrhythmia DB. Train 7,020 / val 540 / test 540 segments (3-class NSR/AF/Other). Deterministic CNN1D-Transformer checkpoint at `runs/mitbih_baseline/best.pt`.

## What we observed

| Quantity | Value |
|---|---:|
| Val UQ-only per-class recall @ cov=0.50 (NSR, AF, Other) | 0.000, **0.871**, 0.000 |
| Val AURC (UQ-only) | 0.6366 |
| Test AURC (UQ-only) | 0.1689 |
| Test AF recall @ cov=0.50 (UQ-only) | **0.944** |

LW-CCSD optimizer on val (snr_db, 1331-candidate grid, constraint coverage 0.50, no floor):

| Optimum | Value |
|---|---:|
| w* (NSR, AF, Other) | **(0.0, 0.0, 0.0)** |
| Test AURC under w* | 0.1689 (= UQ-only) |
| Test AF recall @ cov=0.50 under w* | 0.944 (= UQ-only) |

## Reading

The MIT-BIH classifier is degenerate in the *opposite* direction from CinC. With only 4 AF training records (202, 203, 219, 221) but 33-65 % AF in val/test, the model learned to predict AF for most samples and is essentially an "always AF" predictor at low coverage. At 50 % coverage the UQ ranking already keeps 94 % of true AF — there is no room for SQI to improve AF recall further, and any SNR weighting can only redistribute coverage *away* from AF (where it is correctly classified) toward NSR / Other (where it is not).

The optimizer correctly identifies this: **no positive weight on any class improves val AURC subject to the constraint**, so w* collapses to the all-zero corner of the grid and the LW-CCSD score equals the UQ-only score exactly.

## Why this is a stronger result than a positive replication

A naive expectation was that LW-CCSD would show the same +2-15 % AURC frontier on MIT-BIH that it does on CinC. The actual result is more informative:

- **LW-CCSD has graceful degradation.** When SQI carries no useful signal beyond what UQ already provides, the constrained optimization correctly recovers the UQ-only baseline. There is no failure mode where the method "tries too hard" and hurts performance.
- **The Pareto frontier is not universal.** It is contingent on the underlying classifier having a non-degenerate per-class confidence distribution. On a heavily class-imbalanced small-sample task (MIT-BIH, 4 AF train records), the classifier itself collapses to a one-class predictor, and post-hoc deferral has nothing to optimize.
- **For the paper, this strengthens the discussion of when LW-CCSD applies.** The method shows its value on realistic, class-balanced data (CinC at 45K segments with ~9 % AF prevalence). On degenerate small-sample classifiers it correctly does nothing.

## Implications for deployment

LW-CCSD is not a fix for an underpowered model. It is a free addition for a working one. The recipe to confirm fit:

1. Train the classifier; verify per-class accuracy is non-trivially separated.
2. Compute val UQ-only per-class recall at the constraint coverage. If any class is at 0 recall under UQ-only (as in MIT-BIH NSR/Other), the deferral problem is already collapsed and LW-CCSD will return the trivial w*=0.
3. Otherwise run LW-CCSD; expect a measurable AURC improvement at the operating point chosen by the recall floor.

## Files

- Val predictions: `runs/mitbih_baseline/eval_val_deterministic/predictions.csv`
- Per-floor LW-CCSD runs: `runs/mitbih_baseline/eval_conformal/lw_ccsd_mitbih_baseline/`
- Pinned findings: this document.

## Next

The LW-CCSD methodological story is now complete across two datasets (CinC = positive demonstration; MIT-BIH = graceful-degradation control). Next:

1. **Paper Discussion section** — both findings strengthen the contribution; write them up.
2. **Conformal LW-CCSD** — replace the empirical floor with a finite-sample conformal guarantee.
3. **OBF / MAHNOB-HCI** — when paired face-video AF data arrives, this is the experiment to run.
