# rppg-selective-arrhythmia

**Selective prediction with calibrated uncertainty for contactless arrhythmia detection from facial video (rPPG).**

This repository benchmarks five uncertainty-quantification (UQ) methods on remote photoplethysmography (rPPG) signals derived from public face-video datasets, with the goal of producing a clinically deployable atrial-fibrillation (AF) screen that *defers* to a clinician when its confidence is insufficient.

> **Status — research in progress.** Pipeline implemented; results pending. See [Findings](#findings) for the current state of empirical results, and [Roadmap](#roadmap) for what is still outstanding.

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

1. **First selective-prediction framework** for contactless arrhythmia detection from facial video.
2. **Benchmark of five UQ methods** on rPPG-derived AF classification — MC Dropout, Deep Ensembles, Evidential Deep Learning, SNGP, and Conformal Prediction — with risk-coverage and calibration metrics absent from prior work.
3. **Signal-quality-aware deferral**: a deferral policy that combines model uncertainty with a physical signal-quality estimate (spectral SNR and template SQI), exploiting information that prior rPPG-AF work discards.
4. **Open-source pipeline** released under MIT — data loaders, classical and learned rPPG extractors, classifier, five UQ heads, and selective-evaluation utilities.

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

- [x] Repository scaffolded; package layout, configs, scripts, docs.
- [x] Classical rPPG extractors: CHROM, POS.
- [x] MCD-rPPG dataset loader (`db.csv` index + ECG / PPG / video / sync loaders).
- [x] MediaPipe-based face-ROI extraction (forehead + bilateral cheeks).
- [x] Windowed HR estimation (per-10 s windows, median aggregation), with per-clip HR std as a deferral feature.
- [x] MCD-rPPG validation at scale (n=100): POS MAE 12.39 bpm vs PPG-sync GT ([`docs/baselines/mcd_rppg_v2/`](docs/baselines/mcd_rppg_v2/)).
- [x] Signal-quality metrics: spectral SNR, template SQI.
- [x] 1D-CNN + Transformer classifier.
- [x] Selective metrics: risk-coverage, AURC, ECE, Brier, predictive entropy.
- [x] UQ heads: MC Dropout (working), Deep Ensembles (working), Conformal Prediction (working). Evidential DL module implemented (training-time wiring pending). SNGP scaffolded only.
- [x] Three UQ methods compared on MIT-BIH and synth-rPPG. Pinned in [`docs/baselines/mitbih_uq_v1/`](docs/baselines/mitbih_uq_v1/) and [`docs/baselines/synth_rppg_v1/`](docs/baselines/synth_rppg_v1/).
- [x] MIT-BIH → synthetic rPPG synthesis pipeline (R-peak detection + beat template + 30 Hz downsample + noise model). Pinned in [`docs/baselines/synth_rppg_v1/`](docs/baselines/synth_rppg_v1/).
- [x] CinC 2017 AF Challenge → synth-rPPG at 14× the MIT-BIH AF training scale (8,244 records, 45,064 segments). Three UQ methods compared; all well-calibrated. Pinned in [`docs/baselines/synth_rppg_cinc_v1/`](docs/baselines/synth_rppg_cinc_v1/).
- [x] Signal-quality-aware deferral policy.
- [x] Unit tests for selective metrics and conformal prediction.
- [x] MIT-BIH downloader and PyTorch `Dataset`; subject-disjoint splits.
- [x] Training loop with class-weighted CE, AdamW + cosine, early stopping, W&B optional, checkpointing.
- [x] Selective-evaluation script producing `results.json`, `risk_coverage.csv`, `reliability.csv`, `predictions.csv`.
- [x] First end-to-end run on MIT-BIH (open-data scaffold) — two configurations logged in [`docs/baselines/`](docs/baselines/). Both produced sub-random test accuracy; failures used to inform downstream design (see [Findings](#findings)).
- [ ] Per-split class-coverage verification utility (catch the v0 failure mode before training).
- [ ] Focal-loss / progressive-resampling training option (replacing CE-weight scalar).
- [ ] MediaPipe-based face-ROI extraction wired into the dataset pipeline.
- [ ] Scale MCD-rPPG validation to all 600 subjects (closes remaining gap to paper's 3.80 bpm POS baseline).
- [ ] PhysNet learned rPPG extractor — next-step contender for the high-HR sub-harmonic-lock failure mode.
- [ ] MCD-rPPG dataset adapter for the classifier training loop.
- [ ] SNGP head — spectral-normalized backbone + random-feature GP.
- [ ] MCD-rPPG dataset loader (layout-specific code finalized after download).
- [ ] Phase-1 results on rPPG-10 (independent extractor validation).
- [ ] Phase-1 classifier training on MCD-rPPG healthy-cohort pulse waveforms.
- [x] OBF data access request submitted.
- [ ] MAHNOB-HCI access request (parallel face-video AF channel).
- [ ] Phase-2 results on OBF (AF classification) — pending data access.
- [ ] CinC 2017 AF Challenge dataset → synth-rPPG at 5–10× current AF training scale. **DONE** — see [`docs/baselines/synth_rppg_cinc_v1/`](docs/baselines/synth_rppg_cinc_v1/).
- [ ] Separate `data_seed` from `model_seed` for clean deep-ensemble methodology.
- [x] Signal-quality-aware deferral evaluated on synth-rPPG CinC substrate (the headline methodological claim). **SNR-weighted deferral beats UQ-only by 18 % AURC.** Pinned in [`docs/baselines/sqi_deferral_v1/`](docs/baselines/sqi_deferral_v1/).
- [ ] Evidential Deep Learning training-time integration.
- [ ] SNGP — spectral-normalized backbone + random-feature GP head.
- [ ] Demographic-stratified evaluation.
- [ ] arXiv preprint; submission to IEEE J-BHI.

## How to cite

If this repository contributes to your research, please cite the project repository. A preprint and journal version will be added here when available.

```bibtex
@misc{khan2026rppg_sa,
  author       = {Khan, Muhammad Shahnawaz},
  title        = {{rppg-selective-arrhythmia}: Selective prediction with calibrated uncertainty for contactless arrhythmia detection from facial video},
  year         = {2026},
  howpublished = {\url{https://github.com/ShahnawazKakarh/rppg-selective-arrhythmia}},
  note         = {Work in progress.}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is included at the repository root so GitHub's "Cite this repository" button works directly.

When the accompanying paper appears on arXiv, the entry above will be updated to point to it.

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
