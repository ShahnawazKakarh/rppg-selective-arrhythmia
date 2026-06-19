"""Per-class breakdown of SNR-deferral gain.

Clinically critical question: does combining model confidence with SNR
preferentially preserve AF predictions (good — high-stakes class), or does
the AURC improvement come from disproportionately dumping Other / NSR?

For two scoring functions (UQ-only at w=0 vs SQI-augmented at user-chosen w)
and a sweep of coverage levels, this script reports, per class:
    n_total       — class count in the test set (constant)
    n_kept        — kept after deferral at this coverage
    coverage      — n_kept / n_total
    accuracy      — accuracy on kept subset (precision-like)
    recall        — n_kept_correct / n_total (so deferred ones count as missed)

The recall column is the clinically interpretable one: at 70 % overall
coverage, how many of the AF cases does each policy still get right?
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rppg_sa.extractors.signal_quality import summarize_quality
from rppg_sa.selective.deferral import combine_score
from rppg_sa.utils.config import load_config


DEFAULT_COVERAGES = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]


def _build_test_signals(cfg: dict[str, Any], run_dir: Path):
    source = cfg["data"]["source"]
    if source == "synth_rppg_cinc":
        from rppg_sa.data.cinc2017_synth_torch import (
            CinCSynthRPPGSegmentDataset,
            subject_disjoint_split,
        )

        ds = CinCSynthRPPGSegmentDataset(
            root=cfg["data"]["root"],
            target_fs=float(cfg["data"]["target_fs"]),
            window_seconds=float(cfg["data"]["window_seconds"]),
            step_seconds=float(cfg["data"].get("step_seconds", cfg["data"]["window_seconds"])),
            cache_dir=cfg["data"].get("cache_dir"),
            synth_seed=int(cfg["data"].get("synth_seed", 42)),
            noise_sigma=float(cfg["data"].get("noise_sigma", 0.05)),
            motion_burst_prob=float(cfg["data"].get("motion_burst_prob", 0.0)),
            lighting_flicker_amp=float(cfg["data"].get("lighting_flicker_amp", 0.0)),
            max_records=cfg["data"].get("max_records"),
        )
    elif source == "synth_rppg":
        from rppg_sa.data.synth_rppg_torch import (
            SynthRPPGSegmentDataset,
            subject_disjoint_split,
        )

        ds = SynthRPPGSegmentDataset(
            root=cfg["data"]["root"],
            record_ids=[str(r) for r in cfg["data"]["all_records"]],
            target_fs=float(cfg["data"]["target_fs"]),
            window_seconds=float(cfg["data"]["window_seconds"]),
            step_seconds=float(cfg["data"].get("step_seconds", cfg["data"]["window_seconds"])),
            cache_dir=cfg["data"].get("cache_dir"),
            synth_seed=int(cfg["data"].get("synth_seed", 42)),
            noise_sigma=float(cfg["data"].get("noise_sigma", 0.05)),
            motion_burst_prob=float(cfg["data"].get("motion_burst_prob", 0.0)),
            lighting_flicker_amp=float(cfg["data"].get("lighting_flicker_amp", 0.0)),
        )
    elif source == "mitbih":
        from rppg_sa.data.mitbih_torch import MITBIHSegmentDataset, subject_disjoint_split

        ds = MITBIHSegmentDataset(
            root=cfg["data"]["root"],
            target_fs=float(cfg["data"]["target_fs"]),
            window_seconds=float(cfg["data"]["window_seconds"]),
        )
    else:
        raise ValueError(f"Unsupported data source: {source}")

    splits_file = run_dir / "splits.json"
    if splits_file.exists():
        with splits_file.open() as f:
            sp = json.load(f)
        test_idx = sp["test_idx"]
    else:
        train_ids = [str(r) for r in cfg["data"]["splits"]["train"]]
        val_ids = [str(r) for r in cfg["data"]["splits"]["val"]]
        test_ids = [str(r) for r in cfg["data"]["splits"]["test"]]
        _, _, test_idx = subject_disjoint_split(ds, train_ids, val_ids, test_ids)

    return [ds.signals[i] for i in test_idx], float(cfg["data"]["target_fs"]), ds.LABEL_NAMES


def _load_predictions_csv(path: Path):
    labels, preds, confs = [], [], []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            preds.append(int(row["pred"]))
            confs.append(float(row["confidence"]))
    return (
        np.array(labels, dtype=np.int64),
        np.array(preds, dtype=np.int64),
        np.array(confs, dtype=np.float64),
    )


def _compute_quality(signals, fs: float, key: str) -> np.ndarray:
    out = np.empty(len(signals), dtype=np.float32)
    for i, seg in enumerate(signals):
        q = summarize_quality(seg, fs=fs)
        out[i] = q[key]
    return out


def _select_top_k(score: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k highest scores."""
    if k >= len(score):
        return np.arange(len(score))
    return np.argpartition(score, -k)[-k:]


def per_class_at_coverage(
    score: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    coverage: float,
    label_names: list[str],
) -> dict[str, dict[str, float]]:
    n = len(score)
    k = max(int(round(coverage * n)), 1)
    kept = _select_top_k(score, k)
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[kept] = True

    out: dict[str, dict[str, float]] = {}
    for c, name in enumerate(label_names):
        m_class = labels == c
        n_total = int(m_class.sum())
        if n_total == 0:
            continue
        m_class_kept = m_class & kept_mask
        n_kept = int(m_class_kept.sum())
        correct = (preds[m_class_kept] == labels[m_class_kept]).sum()
        n_correct = int(correct)
        n_correct_total = int((preds[m_class] == labels[m_class]).sum())
        out[name] = {
            "n_total": n_total,
            "n_kept": n_kept,
            "coverage": n_kept / n_total,
            "accuracy_on_kept": (n_correct / n_kept) if n_kept > 0 else float("nan"),
            "recall_overall": n_correct / n_total,
            "recall_no_defer": n_correct_total / n_total,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--signal-quality", default="snr_db", choices=["snr_db", "template_sqi"])
    p.add_argument("--w", type=float, default=0.70,
                   help="Quality-weight for the SQI-augmented score.")
    p.add_argument("--coverages", type=float, nargs="+", default=DEFAULT_COVERAGES)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    cfg = load_config(args.config)
    test_signals, fs, label_names = _build_test_signals(cfg, args.run_dir)
    labels, preds, model_conf = _load_predictions_csv(args.predictions_csv)
    if len(labels) != len(test_signals):
        raise SystemExit(
            f"length mismatch: predictions={len(labels)} vs test signals={len(test_signals)}"
        )
    print(f"n={len(labels)}  classes={label_names}  fs={fs}")
    print(f"Computing {args.signal_quality} ...")
    sq = _compute_quality(test_signals, fs=fs, key=args.signal_quality)

    score_uq = combine_score(model_conf, sq, quality_weight=0.0)
    score_sqi = combine_score(model_conf, sq, quality_weight=args.w)

    out_dir = args.predictions_csv.parent / f"per_class_sqi_deferral{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_for_csv: list[dict[str, Any]] = []

    for coverage in args.coverages:
        uq = per_class_at_coverage(score_uq, labels, preds, coverage, label_names)
        sqi = per_class_at_coverage(score_sqi, labels, preds, coverage, label_names)

        print(
            f"\n=== coverage = {coverage:.2f}  "
            f"(UQ-only vs SQI w={args.w}, {args.signal_quality}) ==="
        )
        header = f"  {'class':<6}  {'n_total':>7}  {'n_kept UQ':>10} {'n_kept SQI':>11}  " \
                 f"{'acc UQ':>7} {'acc SQI':>8}  {'recall UQ':>10} {'recall SQI':>11}"
        print(header)
        for name in label_names:
            if name not in uq:
                continue
            u, s = uq[name], sqi[name]
            print(
                f"  {name:<6}  "
                f"{u['n_total']:>7d}  "
                f"{u['n_kept']:>10d} {s['n_kept']:>11d}  "
                f"{u['accuracy_on_kept']:>7.3f} {s['accuracy_on_kept']:>8.3f}  "
                f"{u['recall_overall']:>10.3f} {s['recall_overall']:>11.3f}"
            )
            rows_for_csv.append({
                "coverage": coverage,
                "class": name,
                "n_total": u["n_total"],
                "n_kept_uq": u["n_kept"],
                "n_kept_sqi": s["n_kept"],
                "acc_kept_uq": u["accuracy_on_kept"],
                "acc_kept_sqi": s["accuracy_on_kept"],
                "recall_uq": u["recall_overall"],
                "recall_sqi": s["recall_overall"],
                "recall_no_defer": u["recall_no_defer"],
                "delta_recall": s["recall_overall"] - u["recall_overall"],
            })

    csv_path = out_dir / f"per_class_{args.signal_quality}_w{args.w:.2f}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
        writer.writeheader()
        writer.writerows(rows_for_csv)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
