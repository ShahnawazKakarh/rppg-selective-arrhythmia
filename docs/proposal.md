# Research Proposal

## Title

**Selective Prediction with Calibrated Uncertainty for Contactless Atrial-Fibrillation Detection from Facial Video**

## Problem

Atrial fibrillation (AF) is the most common sustained cardiac arrhythmia and a major risk factor for stroke. It is frequently asymptomatic and paroxysmal, which makes opportunistic screening — outside of clinical settings — clinically valuable. Remote photoplethysmography (rPPG), which recovers a pulse waveform from subtle skin-color variations in face video, has emerged as a promising contactless modality. Multiple recent studies (OBF benchmark; Yan et al., *Nature Scientific Reports* 2022) report high point-accuracy for rPPG-based AF detection.

However, **none of these works report risk-coverage curves, calibration error, or principled deferral policies**. rPPG signal quality varies sharply with motion, ambient lighting, skin tone, frame rate, and camera quality. A screening tool that returns a confident prediction on a low-quality signal is not deployable. What is missing is *selective prediction*: a system that abstains when uncertain and defers the case to a clinician or to a contact-ECG follow-up.

## Research questions

1. How well do existing uncertainty-quantification (UQ) methods — MC Dropout, Deep Ensembles, Evidential Deep Learning, SNGP, and Conformal Prediction — calibrate confidence for rPPG-based AF classification?
2. Does combining model uncertainty with a physical signal-quality estimate improve selective-prediction performance over either signal alone?
3. How does selective performance generalize across datasets and demographic shifts (skin tone, age, ambient lighting)?

## Approach

### Pipeline

1. **Face tracking and ROI selection** — MediaPipe FaceMesh, forehead and bilateral-cheek ROIs.
2. **rPPG extraction** — classical baselines (CHROM, POS) and a learned extractor (PhysNet). Each method also produces a signal-quality estimate (SNR, template-matching correlation, motion-artifact score).
3. **Rhythm classifier** — 1D-CNN + Transformer over the reconstructed pulse waveform; multi-class output: Normal Sinus Rhythm, Atrial Fibrillation, Other Arrhythmia.
4. **Uncertainty estimation** — five UQ heads benchmarked on the same backbone.
5. **Selective head** — deferral policy combining a UQ-derived confidence score with the signal-quality estimate.
6. **Evaluation** — risk-coverage curve, AURC, selective accuracy at fixed coverage (0.7, 0.8, 0.9, 0.95, 1.0), Expected Calibration Error, Brier score, with bootstrap confidence intervals.

### Data plan

- **Phase 1 (pipeline validation, healthy subjects):** MCD-rPPG (Hugging Face), rPPG-10 (Mendeley).
- **Phase 2 (AF classification):** OBF (academic access via email request).
- **Phase 3 (fallback if OBF is delayed):** synthesize arrhythmic PPG by morphological transforms over MIT-BIH ECG (PhysioNet), paired with public face-video pulse traces. Honest framing: a methodological contribution under the field's known paired-data scarcity.

### Demographic-shift evaluation

Wherever metadata permits, we report selective performance stratified by skin tone (Fitzpatrick proxy), age band, and lighting condition. The expected result — and a reportable finding either way — is that the selective head defers more often on darker skin tones, which is preferable to silent miscalibration.

## Contributions

1. First selective-prediction framework for contactless arrhythmia detection from facial video.
2. Benchmark of five UQ methods on rPPG-derived AF classification, with risk-coverage and calibration metrics absent from prior work.
3. Signal-quality-aware deferral that integrates physical signal properties with model uncertainty.
4. Open-source pipeline (data loaders, extractors, classifier, UQ heads, selective evaluation) released under MIT.

## Target venue

IEEE Journal of Biomedical and Health Informatics (J-BHI). Backup venues: IEEE Transactions on Biomedical Engineering; *npj Digital Medicine*.

## Risks and mitigations

- **Paired AF face-video data is scarce.** Mitigation: OBF request in parallel with Phase-1 work; MIT-BIH-based synthesis as a methodological fallback.
- **Skin-tone bias in rPPG is well-documented.** Mitigation: stratified evaluation reported explicitly; framed as motivation for the selective head rather than hidden.
- **rPPG-AF is no longer an empty field.** Mitigation: novelty claim is the *selective* layer, not raw AF detection. The benchmark itself — five UQ methods, risk-coverage curves, demographic-stratified deferral — is missing from prior work and is the contribution.

## Timeline (indicative)

- Weeks 1–3: Data download, face-tracking pipeline, classical rPPG extractors.
- Weeks 4–6: PhysNet extractor, classifier training on MCD-rPPG / rPPG-10.
- Weeks 7–9: UQ heads (MC Dropout, Ensembles, Evidential, SNGP, Conformal).
- Weeks 10–12: Selective evaluation, demographic-stratified analysis.
- Weeks 13–14: OBF evaluation (if access granted) or MIT-BIH synthesis fallback.
- Weeks 15–16: Writeup, arXiv preprint, J-BHI submission.
