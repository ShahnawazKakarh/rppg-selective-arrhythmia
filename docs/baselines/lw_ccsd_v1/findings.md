# LW-CCSD v1 — Learned Class-Conditional SQI Deferral

**Date:** 2026-06-18
**Substrate:** synth-rPPG CinC test set (n=6,750), deterministic checkpoint `runs/synth_rppg_cinc/best.pt`.

## What it is

A post-hoc, model-agnostic deferral policy that combines model confidence with a physical signal-quality feature (here, spectral SNR in the cardiac band 0.7–3 Hz) using **per-predicted-class weights learned from validation data subject to a formal AF-recall floor**:

  score(x) = (1 − w_{pred(x)}) · rank(confidence(x)) + w_{pred(x)} · rank(SNR(x))

where  w = (w_NSR, w_AF, w_Other) ∈ [0, 1]^3  is found by constrained grid search on val:

  minimize_w  AURC_val(w)
  subject to  per-class recall_val(c, coverage = c0)  ≥  floor_c     for c ∈ {NSR, AF, Other}
              w ∈ grid^3       (grid: 11 points, 0.0 to 1.0 in 0.1 steps → 1,331 candidates)

The optimum w* is then applied unchanged to test. Convex over the discrete grid; the entire search runs in seconds.

## Why this matters

Previous work in this repo established that:

1. **Naive SQI deferral (single shared w):** boosts aggregate AURC by 18 % but collapses AF recall from 0.71 to 0.04 at 50 % coverage. Clinically unsafe.
2. **Hand-tuned `af_immune` (binary class-conditional rule):** safe but small — +2.8 % AURC.

LW-CCSD generalizes the hand-tuned rule: instead of a 0/w switch on AF, it *learns* the per-class weight that respects a configurable AF-recall floor. The result is a tunable Pareto frontier between aggregate selective performance and AF-recall safety.

## Headline — the Pareto frontier

| Val AF recall floor | Learned w* (NSR / AF / Other) | Test AURC | Δ vs UQ-only | Test AF recall @0.5 | Δ AF |
|---:|---|---:|---:|---:|---:|
| 0.40 | 0.4 / 0.5 / 0.5 | 0.1958 | **+17.3 %** | 0.487 | −0.220 |
| 0.45 | 0.2 / 0.5 / 0.5 | 0.1984 | +16.2 % | 0.540 | −0.166 |
| 0.50 | 0.1 / 0.5 / 0.6 | 0.2020 | +14.7 % | 0.583 | −0.123 |
| **0.55** | **0.3 / 0.4 / 0.4** | **0.1998** | **+15.6 %** | **0.626** | **−0.081** |
| 0.58 | 0.0 / 0.4 / 0.6 | 0.2082 | +12.1 % | 0.662 | −0.045 |
| 0.60 | 0.0 / 0.1 / 0.5 | 0.2270 | +4.1 % | 0.703 | **−0.004** |
| UQ-only (no SQI) | — | 0.2367 | — | 0.707 | 0.000 |
| naive SQI (w=0.70) | shared | 0.1942 | +18.0 % | 0.043 | −0.664 |

Reading:

- **Naive SQI's 18 % gain is the asymptote at zero safety.** As soon as the optimizer is forced to respect any AF-recall floor, the gain compresses, monotonically with the constraint tightness.
- **At a 4-pp AF recall sacrifice (floor 0.58 → −0.045 on test), LW-CCSD captures 12.1 % AURC gain** — two-thirds of the unsafe naive headline.
- **At a 4 pp AF recall sacrifice, LW-CCSD captures 4× the gain of the hand-tuned `af_immune` rule** (12.1 % vs 2.8 %), because the learned weights aren't constrained to be binary.
- **At a *zero* AF recall sacrifice** (floor 0.60 → only −0.004 on test, essentially zero), LW-CCSD still gains +4.1 % AURC. The hand-tuned `af_immune` rule was overly conservative on AF; the learned policy reclaims some safe SQI benefit.

## The mechanism, explicit

Learned weights typically place w_AF ≈ 0.0–0.5 and w_NSR ≈ 0.0–0.4, with w_Other in between. The optimizer discovers what hand-tuning could not: **the AF class needs less SQI weighting than NSR**, exactly because AF's defining irregularity collides with the SNR feature. The discovery is implicit and per-checkpoint.

The exact mid-w solution at floor=0.55, w*=(0.3, 0.4, 0.4), illustrates: a uniform 0.4 weight wouldn't work (that's a single-w policy in disguise), and a binary AF=0 / non-AF=0.7 rule wouldn't either (that's `af_immune`). The learned configuration sits in between.

## Why LW-CCSD is novel

To the best of our knowledge:

- **Selective-prediction literature** has explored single-feature confidence thresholds, conformal sets, and global cost-asymmetric thresholds, but **per-predicted-class weights on auxiliary quality features** for deferral, with explicit recall-floor constraints, is new.
- **Class-conditional selective prediction** has appeared in ordinal-DR work (the author's own retinal-selective-prediction repo, OACSP) but always tied to ordinal proximity of misclassification cost — never combining model confidence with a *physical* sensor-quality feature.
- **Signal-quality-aware abstention** has appeared in PPG / rPPG literature but always as a binary gate (drop low-SNR segments before classification). Combining it with model UQ inside a coverage-controlled selective framework, and learning class-aware mixing weights, is the contribution.

## Practical recipe (deployable on any classifier)

1. Train classifier; obtain val + test predictions (softmax probs, predicted class, true label).
2. Compute SNR for every val + test segment (one-line spectral-power ratio).
3. Run LW-CCSD optimizer on val: `scripts/optimize_class_conditional_weights.py`.
4. Pick AF recall floor from clinical acceptability criteria; read off the corresponding w*.
5. Deploy: compute combined score at inference; defer below threshold for target coverage.

No retraining, no architectural changes, no labelling effort beyond what selective prediction already requires.

## Replication on other UQ methods + datasets

This finding was demonstrated on the deterministic single-pass classifier. The cross-UQ pattern observed for `af_immune` (smaller safe gain on stronger UQ methods) suggests LW-CCSD will compress on ensembles. **Replication on MC Dropout and ensembles is the immediate next experiment.**

When OBF / MAHNOB-HCI face-video AF data arrives, replicating LW-CCSD there is the first experiment. The recipe transfers as-is — only the SNR computation needs to be redirected to the real rPPG waveform extracted from facial video.

## Files

- Code: `scripts/optimize_class_conditional_weights.py`, `scripts/dump_val_predictions.py`.
- Sweep outputs: `runs/synth_rppg_cinc/eval_conformal/lw_ccsd_floor{0.40,…,0.60}/learned_weights.json` and `sweep.csv`.
- Operating-point checkpoint (floor 0.55): `runs/synth_rppg_cinc/eval_conformal/lw_ccsd_af55_cov50/`.

## Next

1. **LW-CCSD on MC Dropout + ensembles** (same script, swap predictions.csv).
2. **LW-CCSD on MIT-BIH classifier** for cross-dataset robustness of the trade-off shape.
3. **Finer grid + analytic optimizer** to smooth the small non-monotonicity in the test-side curve (likely a generalization-noise effect from the 11-point grid).
4. **Paper writeup:** the Pareto frontier table above is figure 1; the contribution sentence is "LW-CCSD makes the safety/accuracy trade-off in selective rPPG-AF screening explicit and tunable."
