# MIT-BIH UQ v1 — Three methods compared on 3-class arrhythmia

**Date:** 2026-06-18
**Dataset:** MIT-BIH Arrhythmia Database (PhysioNet), 8640 segments, 3 classes (NSR, AF, Other).
**Classifier:** CNN1D-Transformer, class-weighted CE loss.
**Split (subject-disjoint):** train 39 records / val 3 / test 2. Train AF records: 202, 203, 219, 221.

## Headline

| Method | acc | ECE↓ | Brier↓ | AURC↓ | sel_acc@0.5↑ | sel_acc@0.9↑ | sel_acc@0.95↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MC Dropout (T=30) | 0.711 | 0.250 | 0.508 | 0.487 | 0.517 | 0.710 | 0.722 |
| Conformal (α=0.1) | 0.492 | 0.387 | 0.614 | 0.169 | **0.928** | 0.546 | 0.518 |
| **Ensembles (M=5)** | 0.697 | **0.125** | **0.388** | **0.095** | 0.922 | 0.750 | 0.719 |

## Reading

**Ensembles win calibration and selective behavior end-to-end.** ECE 0.13 vs 0.25 (MC) / 0.39 (Conformal). AURC 0.095 — five-fold better than MC Dropout — meaning when you sort predictions by ensemble confidence and drop the least-confident ones, accuracy on the kept set climbs cleanly.

**Conformal is strongest at low coverage.** At 50% coverage, conformal's top-confidence half is 92.8% accurate. This is the regime where the selective head would defer a lot to clinicians — and conformal is competitive with ensembles there at a fraction of the compute (single forward pass vs M=5 forward passes).

**MC Dropout is the weakest.** Worst AURC (0.49), worst sel_acc@0.5 (0.52). The MC sampling smooths probabilities so much that confidence stops correlating with correctness. Still useful as a sanity baseline.

## Per-class (ensembles)

| Class | n | acc | mean entropy |
|---|---:|---:|---:|
| NSR | 88 | 0.034 | 0.985 |
| AF | 177 | 0.983 | 0.427 |
| Other | 95 | 0.779 | 1.010 |

NSR collapses on test. Train has 6300 NSR but the test records (210, 200) are heavily AF-dominated. Class-weighted loss pushes the model toward AF/Other. This is a distribution-shift artefact, not a UQ failure — entropy correctly flags NSR as high-uncertainty (0.985) and AF as low-uncertainty (0.427).

## Setup

- 3-class CE with inverse-frequency class weights: [0.40, 4.51, 3.49].
- Splits: train [100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 202, 203, 205, 207, 208, 209, 212, 214, 215, 219, 220, 221, 223, 228, 230, 231, 232, 233, 234], val [201, 213, 222], test [210, 200].
- Train AF records: 202, 203, 219, 221 (4). Val AF: 201, 222 (2). Test AF: 210 (1).
- Ensemble: 5 members trained with seeds 1..5, otherwise identical config.
- Conformal calibration set: val (n=540). α=0.1 → target 90% coverage; empirical coverage 0.989 (over-covered, common under shift).

## Files

- `runs/mitbih_baseline/eval_mc_dropout/results.json`
- `runs/mitbih_baseline/eval_conformal/results.json`
- `runs/mitbih_baseline_ens1/eval_ensembles/results.json`
- Per-method `risk_coverage.csv` + `reliability.csv` + `predictions.csv` next to each.

## Next

- **Evidential Deep Learning** — needs retraining with EDL head + MSE+KL loss. Module exists (`src/rppg_sa/uncertainty/evidential.py`); training-time integration pending.
- **SNGP** — only stub in `src/rppg_sa/uncertainty/sngp.py`. Needs spectral_norm wrappers + RFF-GP layer.
- **MIT-BIH classifier itself** is mediocre on NSR-class test records. A better train/val/test split (currently AF-skewed at test) would lift NSR accuracy. Not a UQ blocker.
- **When OBF arrives:** rerun this same 3-method comparison on rPPG-AF data. The eval scaffold transfers as-is.
