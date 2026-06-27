# rppg-selective-arrhythmia

**Learned Class-Conditional Signal-Quality Deferral (LW-CCSD) for selective rPPG-based atrial fibrillation screening.**

[![Zenodo DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20901698-blue)](https://zenodo.org/records/20901698)
[![SSRN](https://img.shields.io/badge/SSRN-abstract%206971878-green)](https://papers.ssrn.com/abstract=6971878)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0007--4055--6563-a6ce39)](https://orcid.org/0009-0007-4055-6563)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository introduces and benchmarks **LW-CCSD**, a post-hoc model-agnostic deferral policy for selective prediction in contactless atrial-fibrillation (AF) screening from remote photoplethysmography (rPPG) signals. The method learns per-predicted-class quality weights combining model confidence with spectral signal-to-noise ratio, subject to a configurable per-class recall floor, and produces a tunable Pareto frontier between selective accuracy and clinical AF-recall safety.

> **Status — v1.9.0 paper draft.** Paper PDF: [`paper/lw-ccsd-rppg-af-v1.9.0.pdf`](paper/lw-ccsd-rppg-af-v1.9.0.pdf). **Zenodo v1.8.0 (current):** [doi.org/10.5281/zenodo.20901698](https://doi.org/10.5281/zenodo.20901698). **SSRN:** [abstract 6971878](https://papers.ssrn.com/abstract=6971878) (revised 2026-06-25 to v1.8.0). Earlier Zenodo snapshots: [v1.2.0](https://doi.org/10.5281/zenodo.20818623), [v1.0.0](https://doi.org/10.5281/zenodo.20776347). v1.8.0 spans 5 UQ methods (deterministic, MC Dropout, deep ensembles, EDL, SNGP) + EDL ensemble + full conformal arc (per-class, Bonferroni, Holm, multivariate-beta). EDL claim established at 5 independent levels (empirical, ensembling, conformal, KL annealing schedule, KL→∞ ablation). The KL→∞ ablation refines the mechanism: the no-LW-CCSD-margin property requires KL annealing, not just the Dirichlet head. See [Findings](#findings) and [Roadmap](#roadmap).

---

## Table of contents

- [Motivation](#motivation)
- [Contributions](#contributions)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Datasets](#datasets)
- [Methods](#methods)
- [Findings](#findings)
- [Roadmap](#roadmap)
- [How to cite](#how-to-cite)
- [References](#references)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Motivation

Atrial fibrillation (AF) is the most common sustained cardiac arrhythmia and a major risk factor for stroke. It is frequently asymptomatic and paroxysmal, which makes opportunistic screening — outside of clinical settings — clinically valuable. Remote photoplethysmography (rPPG), which recovers a pulse waveform from subtle skin-color variations in face video, has emerged as a promising contactless modality. Multiple recent studies (OBF benchmark; Yan et al., *Nature Scientific Reports* 2022) report high point accuracy for rPPG-based AF detection.

However, **none of these works report risk-coverage curves, calibration error, or principled deferral policies**. rPPG signal quality varies sharply with motion, ambient lighting, skin tone, frame rate, and camera quality, so a model that returns a confident prediction on a low-quality signal is not deployable. What is missing is *selective prediction*: a system that abstains when uncertain and defers the case to a clinician or a contact-ECG follow-up.

This repository is the empirical study to fill that gap. The framing is three views of the same problem: selective prediction for rPPG AF detection (methodologically), a UQ benchmark on rPPG arrhythmia classification (empirically), and contactless cardiac screening with deferral (clinically).

## Contributions

1. **LW-CCSD (Learned Class-Conditional Signal-Quality Deferral)** — a post-hoc, model-agnostic deferral policy that learns per-predicted-class quality weights with a configurable per-class recall floor. Produces a tunable Pareto frontier between selective accuracy and AF-recall safety. (Headline contribution.)
2. **Conformal LW-CCSD extension** — Clopper–Pearson finite-sample lower bound on val recall yields a distribution-free coverage guarantee on test recall with essentially zero operational penalty.
3. **First selective-prediction framework for contactless arrhythmia detection from facial video.** Three UQ methods (MC Dropout, Deep Ensembles, Conformal Prediction) benchmarked with risk-coverage and calibration metrics, on a 45,064-segment synthetic-rPPG substrate derived from PhysioNet/CinC 2017.
4. **Empirical mechanism for naive-SQI failure** — quantifies why combining model confidence with signal quality at a single shared weight collapses AF recall from 0.71 to 0.04 at 50 % coverage: AF's irregular rhythm is itself a low-SNR signature in the cardiac band. SNR is shown to be a *class proxy* (88 % of AF segments live in the low-SNR tertile).
5. **Two-axis stratification analysis** — SNR tertile and HR tertile, on three UQ sources, confirms LW-CCSD's gain comes from cross-regime re-ordering rather than within-regime improvement, with positive AURC gain in every HR bin.
6. **Continuous-w optimisation** — Nelder–Mead with soft-penalty constraints confirms the grid-based optimum is in the right neighbourhood; the small test-side non-monotonicities are discretisation-and-generalisation artifacts, not optimisation artifacts.
7. **Deployment finding** — LW-CCSD applied to a single-pass deterministic classifier outperforms an unaugmented five-model deep ensemble on test AURC at one-fifth the inference cost. The method substitutes for compute-expensive ensembling.
8. **Open-source pipeline** released under MIT — data loaders, classical rPPG extractors, classifier, three UQ heads, selective-evaluation utilities, paper LaTeX and HTML sources, and figure generation scripts.

## Repository layout

```
configs/         YAML configs for baselines and UQ variants
docs/            Proposal, dataset notes, related work
notebooks/       Exploratory analysis
scripts/         CLI entry points (download, extract, train, eval)
src/rppg_sa/     Library code
  data/          Dataset loaders (MCD-rPPG, rPPG-10, OBF, MIT-BIH)
  extractors/    rPPG signal extraction (CHROM, POS, PhysNet) + signal quality
  models/        1D-CNN + Transformer classifier
  uncertainty/   MC Dropout, Ensembles, Evidential DL, SNGP, Conformal
  selective/     Risk-coverage, AURC, ECE, signal-quality-aware deferral
  utils/         Config loading, seeding
tests/           Unit tests
```

## Installation

Python 3.10+ is required. PyTorch 2.2+ recommended.

```bash
git clone https://github.com/ShahnawazKakarh/rppg-selective-arrhythmia.git
cd rppg-selective-arrhythmia

# Editable install with dev extras (pytest, ruff, black, jupyter).
pip install -e ".[dev]"
```

GPU is recommended for classifier training but not required for the classical rPPG extractors or for evaluating selective metrics.

## Quickstart

The pipeline is split into four stages, each callable from `scripts/`:

```bash
# 1. Download a dataset (MCD-rPPG used here as the primary training source).
python scripts/download_mcd_rppg.py --out data/raw/mcd_rppg

# 2. Extract pulse waveforms from face video.
python scripts/extract_pulse.py --video path/to/clip.mp4 --method chrom --out pulse.npy

# 3. Train the classifier.
python scripts/train_classifier.py --config configs/baseline_physnet.yaml

# 4. Evaluate selective-prediction performance.
python scripts/eval_selective.py \
    --config configs/selective_mcdropout.yaml \
    --checkpoint runs/mcdropout/best.pt
```

Run the test suite (selective metrics + conformal prediction are fully tested; full pipeline tests pending):

```bash
pytest tests/
```

## Datasets

| Dataset | Subjects | Ground truth | Access | Used for |
|---|---|---|---|---|
| **MCD-rPPG** | 600 (3,600 videos) | PPG + ECG + 13 biomarkers | Open (Hugging Face) | Phase 1 — extractor training, pipeline validation |
| **rPPG-10** | 26 | ECG | Open (Mendeley) | Phase 1 — independent extractor validation |
| **OBF (Oulu Bio-Face)** | Healthy + AF patients | PPG + ECG | Academic request | Phase 2 — primary AF benchmark |
| **MIT-BIH Arrhythmia DB** | 47 | ECG with rhythm annotations | Open (PhysioNet) | Phase 3 — fallback synthesis path |

Full download instructions, license notes, and the OBF access-request email template are in [`docs/datasets.md`](docs/datasets.md).

## Methods

**rPPG extraction.** Classical baselines CHROM (de Haan & Jeanne, 2013) and POS (Wang et al., 2017) are implemented in `src/rppg_sa/extractors/`. The learned baseline PhysNet (Yu et al., 2019) is planned. ROI extraction uses MediaPipe FaceMesh over forehead and bilateral cheek regions.

**Signal quality.** Per-segment spectral SNR (dominant peak vs. surrounding noise in the 0.7–4 Hz band) and template SQI (mean per-beat correlation against the median beat template) are computed both as diagnostics and as inputs to the deferral policy.

**Classifier.** A 1D-CNN + Transformer encoder over the reconstructed pulse waveform (`src/rppg_sa/models/cnn1d_transformer.py`). Convolutional front-end captures local beat morphology; the transformer captures the longer-range rhythm structure characteristic of AF — which is, by definition, irregularity *across* beats rather than within them.

**Uncertainty quantification.** Five methods on a shared backbone:

- **MC Dropout** — dropout-at-inference; predictive entropy and Bayesian mutual information.
- **Deep Ensembles** — M independently-trained members; averaged softmax with mutual-information decomposition.
- **Evidential Deep Learning** — Dirichlet head with annealed-KL Type-II MSE loss.
- **SNGP** — spectral-normalized backbone + random-feature GP head. *(Interface scaffolded; implementation in progress.)*
- **Conformal Prediction** — split conformal with finite-sample coverage correction.

**Selective evaluation.** Risk-coverage curves, AURC, selective accuracy at target coverages (0.7 / 0.8 / 0.9 / 0.95 / 1.0), Expected Calibration Error, and multi-class Brier score, with bootstrap confidence intervals.

**Signal-quality-aware deferral.** A linear combination of rank-normalized model confidence and signal-quality score, with the mixing weight as a tunable hyperparameter. Implemented in `src/rppg_sa/selective/deferral.py`.

Full methodology and rationale in [`docs/proposal.md`](docs/proposal.md).

## Findings

### MIT-BIH pipeline-validation baseline (open-data scaffold)

Three configurations were run on MIT-BIH to validate that the data + classifier + UQ + selective-evaluation stack end-to-end produces real, written artefacts. **All three produced sub-random test accuracy and are reported here as methodological baselines, not as performance baselines.** Raw `results.json` for each run is pinned in [`docs/baselines/`](docs/baselines/).

| Config | Window | Weighting | Train F1 → | Test acc | AURC | ECE | NSR / AF / Other acc | Verdict |
|---|---:|:---:|---:|---:|---:|---:|---|---|
| [`v0_30s_weighted`](docs/baselines/mitbih_v0_30s_weighted/results.json) | 30 s | weighted | n/a → 0.61 val | 24.2 % | 0.538 | 0.509 | 31 % / **n/a** / 11 % | Test split had **no AF** — naive split masked behaviour |
| [`v1_10s_unweighted`](docs/baselines/mitbih_v1_10s_unweighted/results.json) | 10 s* | none | 0.31 → 0.24 val | 16.1 % | 0.856 | 0.726 | 100 % / 0 % / 0 % | **Mode collapse** — model never learned (train F1 flat) |
| [`v2_10s_unweighted_fixed`](docs/baselines/mitbih_v2_10s_unweighted_fixed/results.json) | 10 s | none | **0.32 → 0.72** → 0.39 val | 16.7 % | 0.677 | 0.888 | 100 % / 0 % / 2 % | **Failed subject-disjoint generalization** — model overfits train AF, collapses on test |

*<sub>v1 was nominally 10 s but a loader bug truncated 30 s windows to 10 s rather than re-segmenting; the bug is fixed in v2 and in the current codebase.</sub>*

The failures themselves are the finding. Five concrete lessons feed forward into the rPPG-specific design:

1. **Subject-disjoint splits in small cohorts can drop entire classes from a split.** A naive split of MIT-BIH left zero AF in test (v0). The rPPG datasets (MCD-rPPG, OBF) need verified per-split class coverage *before* training begins.
2. **Class-weighted CE on sparse minorities overpredicts the minority; removing it collapses to the majority.** Neither extreme is acceptable. Focal loss or progressive resampling are the right tools, not a CE weight scalar.
3. **Selective prediction cannot rescue an undertrained model (v1).** When the classifier has not learned the task, the uncertainty ranking it produces is also uninformative — abstention provides no benefit.
4. **Subject-disjoint generalization needs cohort size (v2).** With the loader bug fixed and 8,640 segments available, the model successfully overfits training AF morphology (train F1 0.72) yet fails to generalize to three held-out AF records. This is the *cleanest* motivation for moving to rPPG: MCD-rPPG offers 600 subjects, OBF supplies clinically-graded AF — both above the cohort-size threshold that MIT-BIH cannot meet.
5. **An undertrained model can be *confidently* wrong (v2).** Mean predictive entropy on misclassified AF (0.36 nats) is *lower* than on correctly-classified NSR (0.51 nats) — the model is most confident exactly where it is wrong. This is the precise pathology that selective prediction is supposed to detect, and v2's AURC of 0.888 quantifies how badly entropy-based ranking fails when the underlying model is broken. This strengthens the J-BHI framing rather than weakening it: calibration analysis is the contribution, not point accuracy.

MIT-BIH is parked. The next experimental phase targets the actual research path on rPPG data.

### MCD-rPPG — classical extractor validation (frontal webcam)

Three progressive baselines validate the rPPG extractor + ground-truth
pipeline at increasing scale. **The methodological progression itself is
the finding**: small samples can be misleading; published rPPG benchmarks
aggregate HR by windowed median rather than full-clip Welch peak.

| Baseline | n records | HR aggregation | POS MAE vs PPG-sync (bpm) | CHROM MAE | Pinned at |
|---|---:|---|---:|---:|---|
| [`mcd_rppg_v0`](docs/baselines/mcd_rppg_v0/findings.md) | 8 | Single Welch peak | 5.35 (lucky) | 20.08 | First end-to-end run |
| [`mcd_rppg_v1`](docs/baselines/mcd_rppg_v1/results.json) | 100 | Single Welch peak | 27.10 | 28.58 | Reveals v0 was lucky |
| [**`mcd_rppg_v2`**](docs/baselines/mcd_rppg_v2/findings.md) | 100 | **Windowed median 10 s, 0.7–3 Hz band** | **12.39** | 27.24 | **Current best** |

Paper baseline (Egorov et al. 2025, full dataset): POS 3.80 bpm MAE. We are
~3× above that on a 100-record subset, with the remaining error concentrated
on a specific clinically meaningful subset (see below). Face detection holds
at 100 % across all 100 clips. Full per-record breakdowns and discussion in
[`docs/baselines/mcd_rppg_v2/findings.md`](docs/baselines/mcd_rppg_v2/findings.md).

Four concrete findings:

1. **The classical rPPG pipeline works.** Down from 27 bpm to 12 bpm on the
   same 100 clips, purely by switching from full-clip Welch peak to
   per-10-second-window median — the standard aggregation in the rPPG
   literature. No extractor change.
2. **The remaining 12 bpm is structured, not noise.** Errors concentrate on
   post-exercise high-HR clips (>100 bpm). On normal-HR clips, POS lands
   within a few bpm. The high-HR mode is **classical rPPG sub-harmonic
   lock-on** (de Haan & van Leest 2014; Wang 2017), where chrominance
   projection amplifies the 2nd harmonic. PhysNet (learned extractor) is the
   planned next step here.
3. **This is the right substrate for selective prediction.** Failures cluster
   on clinically meaningful clips (high HR, post-exercise) and co-occur with
   detectable signal-quality drops (low spectral SNR, high per-window HR
   std). The deferral policy can use those features directly.
4. **CHROM is unsuitable for this dataset.** Every clip locks to ~50 bpm
   regardless of true HR; median-across-windows does not help. We attribute
   it to the dataset's consistent MPEG-4 decode artefacts interacting with
   CHROM's higher noise sensitivity. POS is the surviving classical baseline.

The biomarker `pulse` column in `db.csv` is *not* a usable ground truth for
the video time window — it is a discrete clinical cuff reading taken before
or after the 3-minute video, and POS MAE against it is 22.08 bpm (10 bpm
worse than against the synchronized PPG). Subject 1107 has biomarker 146 bpm
but PPG-sync 114 bpm during the actual video; subject 9425 has biomarker 116
bpm but PPG-sync 96 bpm. The synchronized `ppg_sync/*.txt` column 1 is the
correct reference.

### MIT-BIH UQ v1 — three UQ methods compared on a working classifier

With MIT-BIH splits rebalanced (4 AF records in train: 202, 203, 219, 221) and class-weighted CE, the classifier reaches 71 % test accuracy and three UQ methods produce real selective behaviour. Full breakdown in [`docs/baselines/mitbih_uq_v1/findings.md`](docs/baselines/mitbih_uq_v1/findings.md).

| Method | acc | ECE↓ | Brier↓ | AURC↓ | sel_acc@0.5↑ | sel_acc@0.95↑ |
|---|---:|---:|---:|---:|---:|---:|
| MC Dropout (T=30) | 0.711 | 0.250 | 0.508 | 0.487 | 0.517 | 0.722 |
| Conformal (α=0.1) | 0.492 | 0.387 | 0.614 | 0.169 | **0.928** | 0.518 |
| **Ensembles (M=5)** | 0.697 | **0.125** | **0.388** | **0.095** | 0.922 | 0.719 |

Ensembles dominate calibration and AURC. Conformal wins low-coverage (its top-50 % most-confident predictions are 93 % accurate, even though its raw accuracy is the lowest of the three). MC Dropout is the weakest UQ ranker. Same ordering carries forward to the synth-rPPG runs below — the relative behaviour of the three methods is consistent across data sources.

### Synth-rPPG CinC v1 — scaled to 8K AF-labelled records

CinC 2017 AF Challenge (8,244 records, 738 AF) synthesized through the same MIT-BIH pipeline gives 45,064 segments — ~14× the MIT-BIH-derived training scale, with stratified record-level splits. **All three UQ methods land within 3 accuracy points and are well-calibrated** (ECE < 0.08). This is the regime where UQ comparison is methodologically meaningful. Full breakdown in [`docs/baselines/synth_rppg_cinc_v1/findings.md`](docs/baselines/synth_rppg_cinc_v1/findings.md).

| Method | acc | ECE↓ | Brier↓ | AURC↓ | sel_acc@0.5↑ | sel_acc@0.95↑ |
|---|---:|---:|---:|---:|---:|---:|
| **MC Dropout** (T=30) | **0.715** | 0.076 | 0.418 | **0.198** | **0.800** | **0.726** |
| Conformal (α=0.1) | 0.699 | **0.056** | 0.439 | 0.237 | 0.763 | 0.713 |
| Ensembles (M=5) | 0.686 | 0.064 | 0.440 | 0.215 | 0.795 | 0.703 |

The ordering of the three UQ methods is **not consistent across data sources** — ensembles dominated on MIT-BIH, MC Dropout dominates here. That's a publishable methodological finding in itself: the "best UQ method" question is dataset-dependent.

### LW-CCSD v1 — Learned Class-Conditional SQI Deferral (headline contribution)

A principled, post-hoc, model-agnostic deferral policy that learns per-predicted-class quality weights w = (w_NSR, w_AF, w_Other) by constrained grid search on validation data, subject to an AF-recall floor. Converts the unsafe naive-SQI rule and the conservative hand-tuned `af_immune` rule into a **tunable Pareto frontier** between aggregate selective performance and AF-recall safety.

Frontier on synth-rPPG CinC test set:

| Policy | Test AURC | Δ vs UQ-only | Test AF recall @0.5 | Δ AF |
|---|---:|---:|---:|---:|
| UQ-only (no SQI) | 0.2367 | — | 0.707 | 0.000 |
| naive SQI (single shared w=0.70) | 0.1942 | +18.0 % | 0.043 | −0.664 |
| `af_immune` (hand-tuned binary, w=0.70) | 0.2300 | +2.8 % | 0.689 | −0.018 |
| **LW-CCSD floor 0.60** (w*=0.0/0.1/0.5) | 0.2270 | **+4.1 %** | **0.703** | **−0.004** |
| **LW-CCSD floor 0.58** (w*=0.0/0.4/0.6) | 0.2082 | **+12.1 %** | 0.662 | −0.045 |
| **LW-CCSD floor 0.55** (w*=0.3/0.4/0.4) | 0.1998 | **+15.6 %** | 0.626 | −0.081 |
| LW-CCSD floor 0.40 (w*=0.4/0.5/0.5) | 0.1958 | +17.3 % | 0.487 | −0.220 |

Full derivation and per-method replication in [`docs/baselines/lw_ccsd_v1/findings.md`](docs/baselines/lw_ccsd_v1/findings.md). The paper's contribution: **"LW-CCSD makes the safety/accuracy trade-off in selective rPPG-AF screening explicit and tunable; the clinician picks the operating point."**

**Deployment punchline:** LW-CCSD on a single-pass deterministic classifier hits test AURC **0.1998** — *better than the 5-model ensemble UQ-only baseline (0.2155)*. A lightweight model with the free post-hoc deferral rule outperforms ensemble inference. The cross-UQ replication shows the same pattern observed for naive and `af_immune` SQI: SQI value scales inversely with UQ informativeness. LW-CCSD on weak UQ recovers the most gain; on strong UQ (ensembles) gain compresses to ~3-5%.

### Signal-quality-aware deferral v1 — the negative finding that motivated LW-CCSD

Combining model confidence with **spectral SNR** beats UQ-only deferral by 18 % on AURC on the synth-rPPG CinC substrate, with the optimum at w ≈ 0.7 (rank-normalized linear combination). Template SQI barely helps. Same single-model checkpoint, no retraining — the gain is purely from a better ranking score. Full breakdown in [`docs/baselines/sqi_deferral_v1/findings.md`](docs/baselines/sqi_deferral_v1/findings.md).

| SQI feature | UQ-only AURC | Best AURC | Δ | Best w | sel_acc@0.5 |
|---|---:|---:|---:|---:|---:|
| template_sqi | 0.2367 | 0.2289 | +0.0078 (3.3 %) | 0.30 | 0.763 → 0.754 |
| **snr_db** | 0.2367 | **0.1942** | **+0.0425 (18.0 %)** | 0.70 | 0.763 → **0.802** |

The gain replicates across UQ methods: MC Dropout (−entropy) drops AURC from 0.198 → 0.189 (+4.4 %); Ensembles (M=5) drops 0.216 → 0.206 (+4.5 %). SNR-deferral magnitude scales inversely with the strength of the underlying UQ confidence — the simpler the model's uncertainty signal, the more SNR-deferral matters. Cleanest payoff is on deterministic single-forward-pass classifiers (the lightweight clinical-screen regime).

This is the empirical claim the paper hinges on: physical signal-quality features carry deferral information the model confidence inherently misses, and a deployed system can add this overnight (no retraining).

**Important caveat** (see [`per_class/findings.md`](docs/baselines/sqi_deferral_v1/per_class/findings.md)): the per-class breakdown shows the aggregate 18 % AURC gain is **class-asymmetric and clinically harmful** — AF recall collapses from 0.71 to 0.04 at 50 % coverage. AF is defined by irregular rhythm which produces low spectral SNR, so SNR-weighted deferral systematically rejects the very signature the model uses to detect AF. **Resolved with a class-conditional rule** (`af_immune`: SQI for non-AF predictions, pure model confidence for predicted AF): AURC improves +2.8 % over UQ-only with AF recall degraded by ≤2 percentage points. The honest size of the SQI win is therefore 2-3 % AURC, an order of magnitude smaller than the naive headline. The paper's contribution is the *mechanism* (class-defining signal collides with the SQI feature) and the *fix* (class-conditional deferral), not the magnitude.

### Synth-rPPG v1 (MIT-BIH-derived, kept as small-scale baseline)

Fallback path while OBF / MAHNOB-HCI access is pending. ECG R-peaks → asymmetric Gaussian beat templates at PTT lag → downsample to 30 Hz → noise + baseline wander. AF rhythm signal survives the round-trip: val macro-F1 reaches **0.64** on the single-model run. The three UQ methods run end-to-end on the new data source with the same code. Test accuracy is limited by the narrow MIT-BIH split (test = {210, 200}); the next milestone is scaling AF training data via CinC 2017. Pinned in [`docs/baselines/synth_rppg_v1/findings.md`](docs/baselines/synth_rppg_v1/findings.md).

| Method | acc | ECE↓ | Brier↓ | AURC↓ |
|---|---:|---:|---:|---:|
| MC Dropout (T=30) | 0.208 | 0.426 | 0.956 | 0.776 |
| Conformal (α=0.1) | 0.275 | 0.382 | 0.927 | 0.751 |
| **Ensembles (M=5)** | 0.308 | **0.242** | **0.756** | 0.737 |

### Phase 1 (rPPG, healthy cohort) — in progress

MCD-rPPG step-classification (resting vs post-exercise) was attempted as a pipeline sanity task. Both POS-derived features (test F1 0.62) and PPG-sync GT features (test F1 0.65) failed to learn the task: HR distributions overlap heavily at 30-second windows because post-exercise HR recovers within 60–90 s. Pipeline end-to-end validated; task itself is intrinsically weak. Pivoted to MIT-BIH UQ comparison and synth-rPPG fallback above.

### Phase 2 (rPPG, AF detection) — pending data access

OBF access request submitted; MAHNOB-HCI request as parallel channel. Synth-rPPG carries the AF methodology while data requests are in flight.

Planned reporting structure (kept here as a placeholder so the eventual content has a fixed home):

- **Baseline performance.** Multi-class accuracy, macro F1, and per-class precision/recall.
- **Selective performance.** Risk-coverage curves with bootstrap CIs; AURC per UQ method; selective accuracy at 70 / 80 / 90 / 95 percent coverage.
- **Calibration.** ECE and Brier score per method, with reliability diagrams.
- **Signal-quality ablation.** Selective performance with `quality_weight ∈ {0, 0.15, 0.3, 0.5, 0.7, 1.0}`, isolating the contribution of the SQI signal.
- **Demographic stratification.** Selective performance stratified by skin-tone band, age, and lighting condition where metadata permits.

## Roadmap

### Done — methodology + paper (v1.0.0 → v1.7.0)

**LW-CCSD core (v1.0.0)**
- [x] UQ heads: MC Dropout, Deep Ensembles, Conformal Prediction (working end-to-end on two ECG datasets + synthetic substrate).
- [x] MIT-BIH UQ v1 — three UQ methods compared on a working classifier.
- [x] MIT-BIH → synth-rPPG synthesis pipeline (R-peak + asymmetric Gaussian beat template + 30 Hz downsample + noise).
- [x] CinC 2017 AF Challenge → synth-rPPG at 14× MIT-BIH AF scale (45,064 segments). Three UQ methods compared, all well calibrated.
- [x] Signal-quality-aware deferral (naive single-shared-w): +18 % aggregate AURC but AF recall collapses 0.71→0.04 at cov=0.5 — the negative finding that motivated LW-CCSD.
- [x] Per-class breakdown of the SQI failure mode (AF-collapse mechanism, SNR distributions).
- [x] `af_immune` hand-tuned class-conditional rule — safe but conservative.
- [x] **LW-CCSD** — Learned Class-Conditional SQI Deferral with AF-recall-floor constraint. Pareto frontier mapped across three UQ sources.
- [x] **Deployment punchline:** LW-CCSD on deterministic classifier (test AURC 0.1998) beats 5-model ensemble UQ-only (0.2155) at 1/5 inference cost.
- [x] **LW-CCSD on MIT-BIH classifier** — cross-dataset robustness check.
- [x] **Conformal LW-CCSD** — Clopper-Pearson per-class LCB; identical operating point at 90 % confidence, 0.6 % rel AURC cost at 95 %.
- [x] **SNR-stratified evaluation** — 88 % of AF in low-SNR tertile (9× over base rate); cross-regime mechanism isolated.
- [x] **HR-stratified evaluation** — 76 % of AF in high-HR bin; AURC improves in every HR tertile.
- [x] **Cross-UQ stratification** — SNR mechanism holds across MC Dropout, Ensembles, deterministic.
- [x] **Continuous-w (Nelder-Mead)** — grid optimum near-optimal; Pareto non-monotonicities are discretisation-and-generalisation, not optimisation, artifacts.
- [x] **Paper draft** — 7 sections, 18 tables (Table 18 extended), 3 figures. WeasyPrint build (`paper/build.py`) producing `paper/lw-ccsd-rppg-af-v1.9.0.pdf`. IEEE LaTeX source at `paper/main.tex`.
- [x] **Clean ensemble methodology (v1.2.0)** — `data_seed`/`model_seed` decoupled in `scripts/train_classifier.py`; retrained with shared split + independent inits. ECE 0.064→0.052, AURC 0.2155→0.2066, LW-CCSD margin +3.2 % → +6.7 %.

**Conformal completion (v1.2.0)**
- [x] **Bonferroni joint conformal coverage** — family-wise P(all per-class recalls ≥ floor) ≥ 1 − α via union bound (`--conformal-joint-bonferroni`). Net cost per-class → family-wise: 2.05 % rel AURC.
- [x] **Holm step-down joint conformal coverage** — strictly more powerful than Bonferroni at the same family-wise α; recovers 0.6 % rel AURC. Net cost: 1.75 % rel AURC.

**UQ method expansion (v1.3.0 → v1.7.0)**
- [x] **Evidential Deep Learning (EDL) — 4th UQ method (v1.3.0).** Single-pass Dirichlet head with annealed-KL Type-II MSE loss. **EDL is the strongest UQ-only baseline (test AURC 0.1725, test acc 0.739)** of all four single-checkpoint methods studied. **LW-CCSD does not help on top of EDL (−2.3 %)** because EDL absorbs signal-quality information into Dirichlet evidence directly. Paper Section 5.12 + Tables 11–12; pinned in `docs/baselines/lw_ccsd_v1/edl/findings.md`.
- [x] **SNGP — 5th UQ method (v1.4.0).** Liu et al. 2020 distance-aware single-pass UQ. **Best test ECE of all 5 methods (0.029)** at single-pass cost. LW-CCSD gives **+3.7 %** on SNGP — **refutes the EDL mechanism prediction** that any input-conditional UQ would absorb SQI. EDL's no-margin behaviour is specific to its evidence-collapse dynamic, not generic distance-awareness. Paper Section 5.13 + Tables 13–14; pinned in `docs/baselines/lw_ccsd_v1/sngp/findings.md`.
- [x] **EDL ensemble M=5 (v1.5.0).** Variance characterisation: per-member AURC 0.1681 ± 0.0027 (1.6 % rel spread). **EDL ensemble is the strongest configuration studied: test AURC 0.1640 (best), test acc 0.747 (best), Brier 0.422 (best)**. LW-CCSD remains structurally negative (−2.4 %) — mechanism is structural, not per-seed. Paper Section 5.14 + Tables 15–16; pinned in `docs/baselines/lw_ccsd_v1/edl_ensemble/findings.md`.
- [x] **EDL conformal extension (v1.6.0).** Per-class, Bonferroni joint, Holm step-down joint at family-wise α=0.10 on EDL. Negative margin survives every mode: per-class −2.00 %, Bonferroni −2.65 %, Holm −2.00 %. Per-class and Holm produce identical optimum on EDL. Paper Section 5.15 + Table 17; pinned in `docs/baselines/lw_ccsd_v1/edl_conformal/findings.md`.
- [x] **EDL KL annealing sensitivity (v1.7.0).** KL ∈ {5, 10, 15, 25} epochs. Negative margin survives every schedule (mean −2.26 % ± 0.49 %, range [−1.68 %, −2.99 %]). UQ-only AURC stable in [0.1680, 0.1736]. Paper Section 5.16 + Table 18; pinned in `docs/baselines/lw_ccsd_v1/edl_kl_sensitivity/findings.md`.
- [x] **EDL KL→∞ ablation (v1.9.0).** Annealing removed entirely (full KL prior from epoch 1). Negative LW-CCSD margin **disappears** in the no-annealing limit: ΔAURC = +0.92 % at zero AF cost. **Refines the Section 5.12 mechanism**: EDL absorbs signal-quality information in-method *only when KL annealing is enabled* — the no-margin property is a controllable training-time consequence, not an architectural one. Paper Section 5.16 + extended Table 18; pinned in `docs/baselines/lw_ccsd_v1/edl_kl_sensitivity/findings_klinf.md`.

**EDL claim now established at four independent levels:** empirical optimisation (Sec 5.12) · variance across init seeds (Sec 5.14) · conformal coverage at α=0.10 (Sec 5.15) · KL annealing schedule (Sec 5.16). No plausible null hypothesis remains within the EDL training framework.

**Infrastructure**
- [x] **Per-split class-coverage verification utility** — `scripts/verify_split_coverage.py` + CI test catches the v0 failure mode (entire class missing from a split) before training.

**External**
- [x] **Zenodo preprint v1.0.0** — [doi.org/10.5281/zenodo.20776347](https://doi.org/10.5281/zenodo.20776347).
- [x] **Zenodo preprint v1.2.0** — [doi.org/10.5281/zenodo.20818623](https://doi.org/10.5281/zenodo.20818623).
- [x] **Zenodo preprint v1.8.0 (current)** — [doi.org/10.5281/zenodo.20901698](https://doi.org/10.5281/zenodo.20901698).
- [x] **SSRN preprint** — [abstract 6971878](https://papers.ssrn.com/abstract=6971878), revised 2026-06-25 to v1.8.0.
- [x] **Preprints.org submission** — awaiting moderator approval.
- [x] **CITATION.cff** with ORCID + Zenodo DOI.

### Up next — external submission (v1.8.0 PDF ready)
- [x] **Zenodo v1.8.0 published** — [10.5281/zenodo.20901698](https://doi.org/10.5281/zenodo.20901698) (2026-06-26).
- [x] **SSRN revised to v1.8.0 PDF** — 2026-06-25.
- [ ] **arXiv endorsement request** in cs.LG or eess.SP; submit LaTeX source.
- [ ] **TechRxiv submission** when their migration completes.
- [ ] **IEEE BSPC submission** after arXiv and TechRxiv are live.

### Methodology extensions (paper v2.0)
- [ ] **Real demographic stratification** — when OBF / MAHNOB-HCI metadata becomes available.

### Data scaling (paper v2.0)
- [ ] **MAHNOB-HCI access request** (parallel face-video AF channel).
- [ ] **Phase-2 results on OBF / MAHNOB-HCI** — pending data access. LW-CCSD is the first experiment to run.
- [ ] **Phase-1 classifier training on MCD-rPPG healthy-cohort pulse waveforms** (revisited if useful for a robustness section).

### rPPG extractor work (separate research line — next paper)
- [ ] **MediaPipe-based face-ROI extraction wired into the dataset pipeline**.
- [ ] **Scale MCD-rPPG validation to all 600 subjects** (currently 100). Closes gap to paper's 3.80 bpm POS baseline.
- [ ] **PhysNet learned rPPG extractor** — for the high-HR sub-harmonic-lock failure mode observed on MCD-rPPG.

### Training infrastructure (low priority)
- [ ] **Focal-loss / progressive-resampling training option** (replacing CE-weight scalar).

## How to cite

If this repository contributes to your research, please cite the Zenodo preprint and the source repository:

```bibtex
@article{khan2026lwccsd,
  author       = {Khan, Muhammad Shahnawaz},
  title        = {Learned Class-Conditional Signal-Quality Deferral for Selective {rPPG}-Based Atrial Fibrillation Screening},
  year         = {2026},
  month        = {jun},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20901698},
  url          = {https://zenodo.org/records/20901698},
  note         = {Preprint v1.9.0 hosted on Zenodo and SSRN (abstract 6971878, revised 2026-06-25). Code at https://github.com/ShahnawazKakarh/rppg-selective-arrhythmia}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is included at the repository root so GitHub's "Cite this repository" button works directly. The DOI resolves to a Zenodo record with the paper PDF and a versioned snapshot. Update entries to point to the journal version (IEEE J-BHI / Elsevier BSPC) when accepted.

## References

Foundational works this project builds on. (The author list is intentionally short — secondary references appear in [`docs/related_work.md`](docs/related_work.md).)

**rPPG extraction**

1. de Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering*, 60(10), 2878–2886.
2. Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
3. Yu, Z., Peng, W., Li, X., Hong, X., & Zhao, G. (2019). Remote heart rate measurement from highly compressed facial videos: an end-to-end deep learning solution with video enhancement. *Proceedings of the IEEE/CVF International Conference on Computer Vision* (ICCV).

**rPPG-based arrhythmia detection**

4. Li, X., Alikhani, I., Shi, J., Seppänen, T., Junttila, J., Majamaa-Voltti, K., Tulppo, M., & Zhao, G. (2018). The OBF database: a large face video database for remote physiological signal measurement and atrial fibrillation detection. *IEEE International Conference on Automatic Face & Gesture Recognition* (FG).
5. Yan, B. P., Lai, W. H. S., Chan, C. K. Y., et al. (2022). Detection of atrial fibrillation from facial videos using consumer-grade cameras. *Scientific Reports*, 12, Article 13176.

**Uncertainty quantification and selective prediction**

6. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. *International Conference on Machine Learning* (ICML).
7. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems* (NeurIPS).
8. Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep learning to quantify classification uncertainty. *Advances in Neural Information Processing Systems* (NeurIPS).
9. Liu, J. Z., Lin, Z., Padhy, S., Tran, D., Bedrax-Weiss, T., & Lakshminarayanan, B. (2020). Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. *Advances in Neural Information Processing Systems* (NeurIPS).
10. Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *Advances in Neural Information Processing Systems* (NeurIPS).
11. Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511.

**Calibration**

12. Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well-calibrated probabilities using Bayesian binning. *AAAI Conference on Artificial Intelligence*.

**Datasets**

13. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgments

This work uses public datasets — MCD-rPPG (Egorov et al.), rPPG-10 (Mendeley), the OBF database (University of Oulu, on academic request), and the MIT-BIH Arrhythmia Database (PhysioNet / MIT Laboratory for Computational Physiology). The author thanks the maintainers of each.

The methodological approach extends prior work on selective prediction with calibrated uncertainty across medical-imaging modalities; see the references list above for the conceptual lineage.
