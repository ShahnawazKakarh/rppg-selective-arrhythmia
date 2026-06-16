# rppg-selective-arrhythmia

**Selective prediction with calibrated uncertainty for contactless arrhythmia detection from facial video (rPPG).**

This repository benchmarks five uncertainty-quantification (UQ) methods on remote photoplethysmography (rPPG) derived from public face-video datasets, with the goal of producing a clinically deployable atrial-fibrillation (AF) screen that *defers* to a clinician when its confidence is insufficient.

---

## Why this work

Existing rPPG-based AF detection papers report point-accuracy numbers on curated cohorts. None of them report **risk-coverage curves, calibration error, or principled deferral policies** — which is precisely what a contactless screening tool needs before it can be deployed. rPPG signal quality varies dramatically with motion, lighting, skin tone, and camera quality, so a model that *knows when it does not know* is not optional; it is the contribution.

The framing of this repo is three views of the same problem:

1. **Selective prediction for rPPG AF detection** — methodologically.
2. **UQ benchmark on rPPG arrhythmia classification** — empirically.
3. **Contactless cardiac screening with deferral** — clinically.

## What's in scope

- rPPG signal extraction from face video (CHROM, POS, PhysNet baselines).
- 1D-CNN + Transformer classifier over reconstructed pulse waveforms.
- Multi-class output: Normal Sinus Rhythm, Atrial Fibrillation, Other Arrhythmia.
- Five UQ methods benchmarked: MC Dropout, Deep Ensembles, Evidential Deep Learning, SNGP, Conformal Prediction.
- Selective evaluation: risk-coverage curves, AURC, selective accuracy at fixed coverage, ECE.
- Signal-quality-aware deferral that combines SNR-based priors with model uncertainty.

## Datasets

Phase 1 (immediately downloadable, healthy subjects — used for rPPG extractor + pipeline validation):

- **MCD-rPPG** — 3,600 synchronized video recordings from 600 subjects with PPG, ECG, and extended biomarkers.
- **rPPG-10** — 26 subjects, 10-minute recordings with synchronized ECG ground truth.

Phase 2 (arrhythmia-labelled, access-controlled — used for AF classification):

- **OBF (Oulu Bio-Face)** — face video with synchronized physiological signals from both healthy subjects and AF patients. Academic access required (see `docs/datasets.md`).

Phase 3 (fallback, fully open):

- **MIT-BIH Arrhythmia Database** (PhysioNet) — for synthesizing arrhythmic PPG waveforms when paired face-video data is unavailable.

See `docs/datasets.md` for download instructions, license notes, and the OBF access-request email template.

## Status

Scaffolded. Pipeline implementation in progress.

## Repository layout

```
configs/         YAML configs for baselines and UQ variants
docs/            Proposal, dataset notes, related work
notebooks/       Exploratory analysis
scripts/         CLI entry points (download, extract, train, eval)
src/rppg_sa/     Library code
  data/          Dataset loaders
  extractors/    rPPG signal extraction (classical + learned)
  models/        Classifier architectures
  uncertainty/   UQ method implementations
  selective/     Risk-coverage, AURC, ECE, deferral policies
  utils/
tests/
```

## Related work

This repo builds on three threads in the literature: classical rPPG (CHROM, POS, PhysNet), deep AF detection from face video (OBF benchmark; Taiwan cohort, *Nature Scientific Reports* 2022), and selective prediction with uncertainty quantification (MC Dropout, Deep Ensembles, Evidential DL, SNGP, Conformal Prediction). See `docs/related_work.md` for the full positioning.

## License

MIT. See `LICENSE`.

## Citation

If this work helps your research, citation details will appear here once the accompanying paper is on arXiv.
