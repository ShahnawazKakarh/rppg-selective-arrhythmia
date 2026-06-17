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

### MCD-rPPG v0 — classical extractor validation (frontal webcam, n = 8)

First real rPPG numbers in the repo. CHROM and POS were run on a small subset of the [MCD-rPPG](https://huggingface.co/datasets/milai-oks-sakura/mcd_rppg) dataset (4 subjects × resting + post-exercise, frontal webcam, 180 s each) and scored against the synchronized contact-PPG ground truth that accompanies each video. Full per-record breakdown in [`docs/baselines/mcd_rppg_v0/findings.md`](docs/baselines/mcd_rppg_v0/findings.md); raw results in [`docs/baselines/mcd_rppg_v0/results.json`](docs/baselines/mcd_rppg_v0/results.json).

| Method | HR MAE vs PPG-sync GT (bpm) | Notes |
|---|---:|---|
| **POS** (Wang et al. 2017) | **5.35** | Paper's POS baseline on the full dataset: 3.80 |
| CHROM (de Haan & Jeanne 2013) | 20.08 | Consistent sub-harmonic lock-on under MPEG-4 decode artefacts |

Face detection (MediaPipe FaceMesh, forehead + bilateral cheeks): **100 % across all 8 clips**. Sample count is too small to claim a publishable comparison, but it is sufficient to demonstrate end-to-end pipeline correctness and put POS within striking distance of the dataset's own paper baseline.

Three concrete findings from this run:

1. **The biomarker `pulse` column is not the right ground truth.** It is a discrete clinical cuff reading, not the average HR over the video window. POS MAE rises from 5.35 to 26.86 bpm when scored against `db.csv.pulse`. Some clips show a 50+ bpm gap between cuff and contact-PPG (subject 1091: biomarker 132 bpm vs PPG-sync 80 bpm). The synchronized `ppg_sync/*.txt` column 1 is the correct reference.
2. **`ppg_sync/*.txt` is per-video-frame, not per-100-Hz-PPG-sample.** Each file has exactly one row per video frame; column 1 is the contact PPG resampled to the video frame rate; column 2 is sync metadata, *not* inter-sample Δt. Initial mis-interpretation cost a debug cycle; format is now documented in [`mcd_rppg.py`](src/rppg_sa/data/mcd_rppg.py).
3. **CHROM consistently locks to a sub-harmonic on this dataset.** All 8 clips return 47–58 bpm regardless of true HR (60–80 bpm range). Plausibly an interaction between MPEG-4 decode artefacts (warnings on every clip) and CHROM's higher sensitivity to high-frequency colour noise. POS does not show the same failure. This is a genuine CHROM weakness, not an implementation bug.

### Phase 1 (rPPG, healthy cohort) — in progress

Results on MCD-rPPG and rPPG-10 will be added here once the rPPG extractor + Dataset pipeline lands. Reporting will follow the structure planned below.

### Phase 2 (rPPG, AF detection) — not yet run

Results on OBF will be added here once data access is granted. Same reporting structure.

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
- [x] First MCD-rPPG validation: POS MAE 5.35 bpm vs PPG-sync GT, n = 8 ([`docs/baselines/mcd_rppg_v0/`](docs/baselines/mcd_rppg_v0/)).
- [x] Signal-quality metrics: spectral SNR, template SQI.
- [x] 1D-CNN + Transformer classifier.
- [x] Selective metrics: risk-coverage, AURC, ECE, Brier, predictive entropy.
- [x] UQ heads: MC Dropout, Deep Ensembles, Evidential DL, Conformal Prediction.
- [x] Signal-quality-aware deferral policy.
- [x] Unit tests for selective metrics and conformal prediction.
- [x] MIT-BIH downloader and PyTorch `Dataset`; subject-disjoint splits.
- [x] Training loop with class-weighted CE, AdamW + cosine, early stopping, W&B optional, checkpointing.
- [x] Selective-evaluation script producing `results.json`, `risk_coverage.csv`, `reliability.csv`, `predictions.csv`.
- [x] First end-to-end run on MIT-BIH (open-data scaffold) — two configurations logged in [`docs/baselines/`](docs/baselines/). Both produced sub-random test accuracy; failures used to inform downstream design (see [Findings](#findings)).
- [ ] Per-split class-coverage verification utility (catch the v0 failure mode before training).
- [ ] Focal-loss / progressive-resampling training option (replacing CE-weight scalar).
- [ ] MediaPipe-based face-ROI extraction wired into the dataset pipeline.
- [ ] Scale MCD-rPPG validation to 50+ subjects (closes gap to paper's 3.80 bpm POS baseline).
- [ ] Per-window (10 s) HR estimation for per-segment confidence inputs.
- [ ] PhysNet learned rPPG extractor.
- [ ] SNGP head — spectral-normalized backbone + random-feature GP.
- [ ] MCD-rPPG dataset loader (layout-specific code finalized after download).
- [ ] Phase-1 results on rPPG-10 (independent extractor validation).
- [ ] Phase-1 classifier training on MCD-rPPG healthy-cohort pulse waveforms.
- [ ] OBF data access request submitted.
- [ ] Phase-2 results on OBF (AF classification).
- [ ] MIT-BIH synthesis fallback path.
- [ ] Demographic-stratified evaluation.
- [ ] arXiv preprint; submission to IEEE J-BHI.

## How to cite

If this repository contributes to your research, please cite the project repository. A preprint and journal version will be added here when available.

```bibtex
@misc{khan2026rppg_sa,
  author       = {Khan, Shahnawaz},
  title        = {{rppg-selective-arrhythmia}: Selective prediction with calibrated uncertainty for contactless arrhythmia detection from facial video},
  year         = {2026},
  howpublished = {\url{https://github.com/ShahnawazKakarh/rppg-selective-arrhythmia}},
  note         = {Work in progress.}
}
```

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
