# Baselines log

This directory pins the raw results of each experimental run. The intent is a transparent record of what was tried, what happened, and what we learned — including failures. Each subdirectory contains a `results.json` with config + metrics + verdict.

## Index

| ID | Date (approx) | Source | Key config | Headline | Verdict |
|---|---|---|---|---|---|
| [`mitbih_v0_30s_weighted`](mitbih_v0_30s_weighted/results.json) | 2026-06-17 | MIT-BIH | 30 s windows, class-weighted CE, dropout 0.2 | Test acc 24.2%, AURC 0.538 | **Failed split** — test set contained no AF samples |
| [`mitbih_v1_10s_unweighted`](mitbih_v1_10s_unweighted/results.json) | 2026-06-17 | MIT-BIH | 10 s windows (loader bug: effectively 30 s truncated), no class weighting, dropout 0.4 | Test acc 16.1%, AURC 0.856 | **Mode collapse** — model never learned (train F1 flat) |
| [`mitbih_v2_10s_unweighted_fixed`](mitbih_v2_10s_unweighted_fixed/results.json) | 2026-06-17 | MIT-BIH | True 10 s windows (loader bug fixed), no class weighting, dropout 0.4 | Test acc 16.7%, AURC 0.888 | **Failed subject-disjoint generalization** — train F1 0.72 → test collapse to majority class |

## What we learned

All three attempts produced sub-random test accuracy on the 3-class task (NSR / AF / Other) and are therefore not useful as performance baselines. They are, however, useful as **methodological** baselines — each exposes a distinct failure mode that the rPPG pipeline must avoid:

1. **Subject-disjoint splits in small cohorts can drop entire classes from a split.** v0 left zero AF in test, silently masking model behaviour. The rPPG datasets (MCD-rPPG, OBF) need verified per-split class coverage *before* training begins.
2. **Class-weighted CE on sparse minorities overpredicts the minority (v0); removing it collapses to the majority (v1, v2).** Neither extreme works. Focal loss or progressive resampling are the right tools, not a CE weight scalar.
3. **Selective prediction cannot rescue an undertrained model (v1).** When the classifier has not learned the task, the uncertainty ranking it produces is also uninformative — abstention provides no benefit.
4. **Subject-disjoint generalization needs cohort size (v2).** With the loader bug fixed and 8,640 segments available, the model successfully overfits the training AF morphology (train F1 0.72) yet fails to generalize to three held-out AF records. This is the *cleanest* motivation for moving to rPPG: MCD-rPPG offers 600 subjects, OBF supplies clinically-graded AF — both above the cohort-size threshold MIT-BIH cannot meet.
5. **An undertrained model can be confidently wrong (v2).** Notably, the v2 model's mean predictive entropy on misclassified AF (0.36 nats) is *lower* than on correctly-classified NSR (0.51 nats) — i.e. it is most confident exactly where it is wrong. This is the precise pathology that selective prediction is supposed to detect, and the v2 AURC of 0.888 quantifies how badly entropy-based ranking fails when the underlying model is broken. Both observations strengthen the J-BHI framing: calibration analysis is the contribution, not point accuracy.

## Why this is reported anyway

A J-BHI selective-prediction paper that only includes successful runs is suspect. Pinning the failed configurations makes the eventual successful rPPG result reproducible and verifiable: reviewers can see exactly what didn't work and confirm that the published result isn't cherry-picked.

## Provenance

Each `results.json` contains the exact config that produced it. The full training history and per-sample predictions (`history.json`, `predictions.csv`) for each run live under `runs/` in the local workspace and are not tracked in git (see `.gitignore`). They are reproducible from the configs.

## Next runs

MIT-BIH baseline is parked. Next experiments target the actual research path:
1. MCD-rPPG ingestion and rPPG extractor validation (CHROM / POS on healthy-cohort heart-rate estimation).
2. PhysNet learned extractor.
3. OBF AF detection (pending data-access response).
