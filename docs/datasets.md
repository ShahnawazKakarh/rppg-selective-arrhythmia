# Datasets

## Phase 1 — immediately downloadable, healthy subjects

### MCD-rPPG
- **Source:** Hugging Face — `milai-oks-sakura/mcd_rppg` (mirror of the original `kyegorov/mcd_rppg`, which now 404s).
- **Companion repo:** https://github.com/ksyegorov/mcd_rppg
- **Size:** 3,600 synchronized video recordings from 600 subjects. ~135 GB total.
- **Ground truth:** 100 Hz PPG + ECG + 13 health biomarkers (BP, SpO2, glucose, temperature, respiratory rate, stress level, age, sex, BMI, etc.).
- **Conditions:** resting and post-exercise; three camera views (frontal webcam, FullHD camcorder, mobile phone).
- **License:** CC-BY-4.0, but **gated** — you must visit the dataset page while signed in to HF, click *Agree and access repository*, and accept sharing your email + HF username with the dataset authors. After agreeing, your token also needs `milai-oks-sakura/mcd_rppg` in its allow-list (a Classic Read token covers this automatically; a Fine-grained token needs the repo added explicitly).
- **Use here:** primary training set for the rPPG extractor and for pipeline validation.

### rPPG-10
- **Source:** Mendeley Data — DOI `10.17632/bx8982xgwt.1`
- **Size:** 26 Portuguese university students (one excluded for artifacts), mean age 22.5 ± 1.2.
- **Ground truth:** synchronized ECG.
- **Conditions:** 10-minute recordings, three ROIs (forehead, left cheek, right cheek), natural lighting.
- **License:** as listed on the Mendeley record.
- **Use here:** independent validation set for the rPPG extractor.

## Phase 2 — arrhythmia-labelled, access-controlled

### OBF (Oulu Bio-Face)
- **Source:** Center for Machine Vision and Signal Analysis, University of Oulu, Finland.
- **Contents:** face video with synchronized PPG and ECG, from both healthy subjects and patients diagnosed with atrial fibrillation. Canonical benchmark for rPPG-based AF detection.
- **Access:** academic request with signed end-user license agreement. Procedure handled out-of-band; no recipient or template details in this repo.
- **Use here:** primary AF/NSR/Other classification benchmark.

## Phase 3 — fallback (fully open)

### MIT-BIH Arrhythmia Database
- **Source:** PhysioNet — https://physionet.org/content/mitdb/
- **Contents:** 48 half-hour two-channel ECG recordings, annotated for arrhythmia events including AF.
- **License:** PhysioNet Credentialed Health Data License / Open Data Commons (see record).
- **Use here:** synthesize arrhythmic PPG waveforms (morphology transforms over ECG R-R intervals) when paired face-video AF data is unavailable. Framed as a methodological contribution given the field's paired-data scarcity.

## Storage layout

```
data/
├── raw/
│   ├── mcd_rppg/
│   ├── rppg10/
│   ├── obf/                  # gitignored, access-controlled
│   └── mitbih/
├── processed/
│   ├── pulse_waveforms/      # extracted rPPG signals
│   └── segments/             # 30-s windows ready for classifier
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Everything under `data/` is gitignored. Provenance files (download timestamps, source URLs, checksums) live in `data/raw/<dataset>/PROVENANCE.txt`.
