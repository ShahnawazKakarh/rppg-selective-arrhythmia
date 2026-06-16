# Related Work

Positioning of this repository against three threads in the literature.

## 1. Classical and learned rPPG extraction

- **CHROM** (de Haan and Jeanne, 2013) — chrominance-based skin reflection model; the standard classical baseline.
- **POS** (Wang et al., 2017) — plane-orthogonal-to-skin projection; outperforms CHROM under motion.
- **PhysNet** (Yu et al., 2019) — end-to-end 3D-CNN for pulse extraction from face video.
- **RhythmNet** (Niu et al., 2019) — spatial-temporal map regression for heart-rate estimation, evaluated on VIPL-HR.

**Position of this work:** uses CHROM and POS as classical baselines and PhysNet as a learned reference. Extraction is *not* the contribution; we treat it as a fixed front-end.

## 2. rPPG-based atrial-fibrillation detection

- **OBF database** (Li et al., 2018) — first public face-video benchmark including AF patients alongside healthy subjects. Establishes that rPPG can recover the irregular inter-beat intervals characteristic of AF.
- **Yan et al., *Nature Scientific Reports* 2022** — 453 participants classified by 12-lead ECG into AF / NSR / Other; deep CNN over rPPG segments with majority voting at the patient level. Reports point accuracy but no risk-coverage or calibration analysis.
- **Fusing-subtle-variations work (IEEE TIM / IEEE TBME line)** — lightweight CNN with attention modules for AF detection; 7,216 segments from 452 subjects across AF, NSR, and Other categories.

**Position of this work:** rPPG-based AF detection is no longer an empty field. Our contribution is *not* "first to detect AF from face video." It is the selective-prediction layer — risk-coverage analysis, calibration, principled deferral, and demographic-stratified evaluation — none of which appears in the above works.

## 3. Selective prediction and uncertainty quantification

- **MC Dropout** (Gal and Ghahramani, 2016) — dropout-at-inference for cheap epistemic uncertainty.
- **Deep Ensembles** (Lakshminarayanan et al., 2017) — strong calibration baseline; expensive but reliable.
- **Evidential Deep Learning** (Sensoy et al., 2018; Amini et al., 2020) — single-forward-pass Dirichlet-prior uncertainty.
- **SNGP** (Liu et al., 2020) — spectral-normalized neural Gaussian process; competitive single-pass alternative to ensembles.
- **Conformal Prediction** (Vovk; Angelopoulos and Bates, 2021) — distribution-free prediction sets with finite-sample coverage guarantees.
- **Selective classification with deep models** (Geifman and El-Yaniv, 2017) — risk-coverage as the natural evaluation framework.
- **AURC and selective metrics** (Geifman et al., 2018; Ding et al., 2020) — formal evaluation of abstention.

**Position of this work:** these methods have been benchmarked on standard image classification (CIFAR, ImageNet) and on a few medical-imaging settings (retinal, dermatology), but not on rPPG-derived arrhythmia classification. Signal-quality estimates from rPPG provide a natural complement to model uncertainty, which is the technical novelty.

## Methodologically adjacent work in this author's portfolio

- **retinal-selective-prediction** — same five-UQ-method spine applied to diabetic-retinopathy grading from fundus images. Establishes the framework that this repo extends to a new modality.
- **speech-emotion-recognition-transfer-learning** — cross-attention multimodal fusion; the architectural pattern is reusable here for fusing pulse waveform with signal-quality features.

The three repos together form a coherent line: *selective prediction with calibrated uncertainty for accessible medical AI*, across retinal, cardiac, and speech-affective modalities.
