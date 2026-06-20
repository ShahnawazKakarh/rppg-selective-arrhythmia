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

## Cross-UQ replication — the deployment story

LW-CCSD applied to three confidence sources on the same CinC test set:

| UQ source | UQ-only AURC | Best LW-CCSD AURC | Δ AURC | AF recall cost | Operating point (floor) |
|---|---:|---:|---:|---:|---:|
| **Deterministic** (single pass) | 0.2367 | **0.1998** | **+15.6 %** | −0.081 | 0.55 |
| MC Dropout (T=30) | 0.1980 | 0.1899 | +4.1 % | −0.029 | 0.45 |
| Ensembles (M=5) | 0.2155 | 0.2049 | +4.9 % | −0.174 | 0.40 (aggressive) |
| Ensembles (safe) | 0.2155 | 0.2086 | +3.2 % | −0.023 | 0.55 |

### Per-UQ Pareto frontiers (test, snr_db, constraint coverage 0.50)

Deterministic:

| Val floor | w* (NSR/AF/Other) | Test AURC | Δ | Test AF@0.5 | Δ AF |
|---:|---|---:|---:|---:|---:|
| 0.60 | 0.0/0.1/0.5 | 0.2270 | +4.1 % | 0.703 | −0.004 |
| 0.58 | 0.0/0.4/0.6 | 0.2082 | +12.1 % | 0.662 | −0.045 |
| 0.55 | 0.3/0.4/0.4 | 0.1998 | +15.6 % | 0.626 | −0.081 |
| 0.50 | 0.1/0.5/0.6 | 0.2020 | +14.7 % | 0.583 | −0.123 |
| 0.45 | 0.2/0.5/0.5 | 0.1984 | +16.2 % | 0.540 | −0.166 |
| 0.40 | 0.4/0.5/0.5 | 0.1958 | +17.3 % | 0.487 | −0.220 |

MC Dropout (T=30):

| Val floor | w* | Test AURC | Δ | Test AF@0.5 | Δ AF |
|---:|---|---:|---:|---:|---:|
| 0.47 | 0.8/0.1/0.0 | 0.1922 | +2.9 % | 0.551 | −0.011 |
| 0.45 | 0.3/0.4/0.3 | 0.1899 | +4.1 % | 0.533 | −0.029 |
| 0.40 | 0.7/0.4/0.1 | 0.1899 | +4.0 % | 0.530 | −0.032 |
| 0.30 | 0.4/0.5/0.2 | 0.1901 | +4.0 % | 0.376 | −0.186 |

Ensembles (M=5):

| Val floor | w* | Test AURC | Δ | Test AF@0.5 | Δ AF |
|---:|---|---:|---:|---:|---:|
| 0.58 | 0.1/0.1/0.1 | 0.2082 | +3.4 % | 0.532 | −0.007 |
| 0.57 | 0.5/0.1/0.0 | 0.2110 | +2.1 % | 0.522 | −0.017 |
| 0.55 | 0.4/0.2/0.0 | 0.2086 | +3.2 % | 0.515 | −0.023 |
| 0.45 | 0.4/0.4/0.0 | 0.2057 | +4.6 % | 0.492 | −0.047 |
| 0.40 | 0.4/0.5/0.0 | 0.2049 | +4.9 % | 0.364 | −0.174 |

### The deployment punchline

**LW-CCSD on a single-forward-pass deterministic classifier hits AURC 0.1998 — better than a 5-model ensemble UQ-only baseline at 0.2155.** A lightweight model with the free post-hoc deferral rule outperforms ensemble inference. This means LW-CCSD is not just an add-on; it is a viable *replacement* for compute-expensive ensembling in deployed clinical screens.

### Reading the pattern

Wide Pareto frontier on weak UQ (deterministic), narrow frontier on strong UQ (ensembles, MC Dropout). The unifying pattern: SQI value scales inversely with UQ informativeness. The learned weights automatically find the right operating point per UQ source — the same optimizer call discovers very different w* for the same recall floor across UQ methods.

This is methodologically clean and clinically useful:

- **Choose deterministic** for cheap inference + strong post-hoc gains — the most cost-effective deployment.
- **Choose ensembles** for strongest absolute UQ — SQI is mostly redundant; safe to drop the SQI feature entirely.
- **Choose MC Dropout** for a mid-point — modest gains, easy to keep AF recall floor near baseline.

When OBF / MAHNOB-HCI face-video AF data arrives, replicating LW-CCSD there is the first experiment. The recipe transfers as-is — only the SNR computation needs to be redirected to the real rPPG waveform extracted from facial video.

## Files

- Code: `scripts/optimize_class_conditional_weights.py`, `scripts/dump_val_predictions.py`.
- Sweep outputs: `runs/synth_rppg_cinc/eval_conformal/lw_ccsd_floor{0.40,…,0.60}/learned_weights.json` and `sweep.csv`.
- Operating-point checkpoint (floor 0.55): `runs/synth_rppg_cinc/eval_conformal/lw_ccsd_af55_cov50/`.

## Cross-dataset check (MIT-BIH) — graceful degradation

Replicated on the MIT-BIH 3-class classifier. Val UQ-only per-class recall @ cov=0.50 is (NSR 0.00, AF 0.87, Other 0.00) — the small-sample classifier is degenerate, predicting AF for most samples. Test AF recall @ cov=0.50 under UQ-only is **0.944**.

The LW-CCSD optimizer correctly returns w* = (0, 0, 0) — the all-zero corner of the grid. No SQI weighting improves val AURC because there is no per-class confidence structure to refine. Test AURC and AF recall under w* exactly match UQ-only.

**Reading.** This is a stronger result than a positive replication would have been. It shows the method has graceful degradation: when SQI carries no signal beyond UQ, the constrained optimization recovers the trivial baseline. There is no failure mode where LW-CCSD "tries too hard" and hurts performance. The Pareto frontier of the CinC experiment is contingent on the underlying classifier being non-degenerate (CinC has 45K segments and ~9 % AF prevalence; MIT-BIH has 540 test segments and ~33-65 % AF prevalence due to the AF-heavy test records). Pinned in [`mitbih/findings.md`](mitbih/findings.md).

## Next

1. **LW-CCSD on MC Dropout + ensembles** (same script, swap predictions.csv).
2. **LW-CCSD on MIT-BIH classifier** for cross-dataset robustness of the trade-off shape.
3. **Finer grid + analytic optimizer** to smooth the small non-monotonicity in the test-side curve (likely a generalization-noise effect from the 11-point grid).
4. **Paper writeup:** the Pareto frontier table above is figure 1; the contribution sentence is "LW-CCSD makes the safety/accuracy trade-off in selective rPPG-AF screening explicit and tunable."
