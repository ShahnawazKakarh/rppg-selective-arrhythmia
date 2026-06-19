"""Class-conditional SQI deferral: defer NSR/Other on low SNR, never penalize AF.

The naive SQI rule combines model confidence with SNR for every prediction.
But AF is *defined* by irregular rhythm, which lowers spectral SNR. So the
naive rule throws away AF predictions whose SNR is naturally low, collapsing
AF recall from 0.71 to 0.04 at 50 % coverage (see per_class/findings.md).

This script evaluates two fixes:

  (A) AF-immune SQI deferral:
        combined(x) = (1 - w) * conf(x) + w * SNR(x)   if pred(x) != AF
                    = conf(x)                          if pred(x) == AF

  (B) Within-class SNR rank-normalization:
        SNR is rank-normalized inside each predicted class before combination.
        This removes the cross-class SNR bias entirely.

Both keep model confidence as the only signal for predicted AF in policy (A),
or remove the cross-class bias in policy (B). Headline metric: AURC subject
to an AF-recall floor (we use >= 0.70).

Compares each fix against (a) UQ-only and (b) naive SQI at the same w.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rppg_sa.data.mitbih import CLASS_TO_INDEX
from rppg_sa.extractors.signal_quality import summarize_quality
from rppg_sa.selective.deferral import combine_score
from rppg_sa.selective.metrics import (
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)
from rppg_sa.utils.config import load_config


DEFAULT_COVERAGES = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
AF_RECALL_FLOOR = 0.70  # clinical safety floor


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
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
        for row in csv.DictReader(f):
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


# ----------------------------------------------------------------------
# Scoring policies
# ----------------------------------------------------------------------
def _rank_normalize(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order.astype(np.float64) / max(len(x) - 1, 1)


def policy_uq_only(conf: np.ndarray, sq: np.ndarray, preds: np.ndarray, w: float, af_idx: int):
    return _rank_normalize(conf)


def policy_naive_sqi(conf: np.ndarray, sq: np.ndarray, preds: np.ndarray, w: float, af_idx: int):
    return combine_score(conf, sq, quality_weight=w)


def policy_af_immune(conf: np.ndarray, sq: np.ndarray, preds: np.ndarray, w: float, af_idx: int):
    """SQI for non-AF predictions; pure confidence for predicted AF."""
    nonaf_score = combine_score(conf, sq, quality_weight=w)
    conf_score = _rank_normalize(conf)
    return np.where(preds == af_idx, conf_score, nonaf_score)


def policy_within_class(conf: np.ndarray, sq: np.ndarray, preds: np.ndarray, w: float, af_idx: int):
    """Rank-normalize SNR INSIDE each predicted class, then combine. Removes
    the cross-class SNR bias: an AF prediction with median-for-AF SNR scores
    the same on the SQI axis as a median-for-NSR NSR prediction."""
    conf_r = _rank_normalize(conf)
    sq_r = np.zeros_like(sq, dtype=np.float64)
    for c in np.unique(preds):
        mask = preds == c
        if mask.sum() == 0:
            continue
        sq_r[mask] = _rank_normalize(sq[mask])
    return (1.0 - w) * conf_r + w * sq_r


POLICIES = {
    "uq_only":      policy_uq_only,
    "naive_sqi":    policy_naive_sqi,
    "af_immune":    policy_af_immune,
    "within_class": policy_within_class,
}


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------
def _select_top_k(score: np.ndarray, k: int) -> np.ndarray:
    if k >= len(score):
        return np.arange(len(score))
    return np.argpartition(score, -k)[-k:]


def per_class_at_coverage(score, labels, preds, coverage, label_names):
    n = len(score)
    k = max(int(round(coverage * n)), 1)
    kept = _select_top_k(score, k)
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[kept] = True
    out = {}
    for c, name in enumerate(label_names):
        m_class = labels == c
        n_total = int(m_class.sum())
        if n_total == 0:
            continue
        m_class_kept = m_class & kept_mask
        n_kept = int(m_class_kept.sum())
        n_correct = int((preds[m_class_kept] == labels[m_class_kept]).sum())
        out[name] = {
            "n_total": n_total,
            "n_kept": n_kept,
            "accuracy_on_kept": (n_correct / n_kept) if n_kept > 0 else float("nan"),
            "recall_overall": n_correct / n_total,
        }
    return out


def evaluate_policy(name, score, labels, preds, correct, coverages, label_names):
    rc = risk_coverage_curve(score, correct)
    sel_acc = {c: selective_accuracy_at_coverage(score, correct, c) for c in coverages}
    per_class = {c: per_class_at_coverage(score, labels, preds, c, label_names) for c in coverages}
    return {
        "name": name,
        "aurc": float(rc.aurc),
        "accuracy_at_full": float(correct.mean()),
        "sel_acc": {f"{c}": float(v) for c, v in sel_acc.items()},
        "per_class": {f"{c}": pc for c, pc in per_class.items()},
    }


def aurc_under_af_floor(score, labels, preds, correct, af_idx, recall_floor):
    """Highest coverage c such that AF recall on top-c subset >= recall_floor,
    and the AURC integrated over coverages [0, c]."""
    n = len(score)
    order = np.argsort(-score)  # high to low
    af_mask = labels == af_idx
    af_total = max(int(af_mask.sum()), 1)
    af_correct_running = 0
    running_correct = 0
    coverages = []
    risks = []
    feasible_cov = 0.0
    for i, idx in enumerate(order):
        running_correct += int(correct[idx])
        if af_mask[idx]:
            af_correct_running += int(correct[idx] and preds[idx] == af_idx)
        c = (i + 1) / n
        coverages.append(c)
        risks.append(1.0 - running_correct / (i + 1))
        if af_correct_running / af_total >= recall_floor:
            feasible_cov = c
    if feasible_cov == 0.0:
        return float("nan"), 0.0
    cov = np.array(coverages)
    rsk = np.array(risks)
    mask = cov <= feasible_cov
    trapz = getattr(np, "trapezoid", np.trapz)
    aurc_floored = float(trapz(rsk[mask], cov[mask]))
    return aurc_floored, float(feasible_cov)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--predictions-csv", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--signal-quality", default="snr_db", choices=["snr_db", "template_sqi"])
    p.add_argument("--w", type=float, default=0.70,
                   help="Quality weight for naive_sqi, af_immune, within_class.")
    p.add_argument("--af-recall-floor", type=float, default=AF_RECALL_FLOOR)
    p.add_argument("--coverages", type=float, nargs="+", default=DEFAULT_COVERAGES)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    cfg = load_config(args.config)
    test_signals, fs, label_names = _build_test_signals(cfg, args.run_dir)
    labels, preds, conf = _load_predictions_csv(args.predictions_csv)
    if len(labels) != len(test_signals):
        raise SystemExit(
            f"length mismatch: predictions={len(labels)} vs test signals={len(test_signals)}"
        )
    print(f"n={len(labels)}  classes={label_names}  fs={fs}")
    print(f"Computing {args.signal_quality} ...")
    sq = _compute_quality(test_signals, fs=fs, key=args.signal_quality)

    af_idx = CLASS_TO_INDEX["AF"]
    correct = (preds == labels).astype(np.float64)

    results = {}
    summary_rows = []
    for name, policy_fn in POLICIES.items():
        score = policy_fn(conf, sq, preds, args.w, af_idx)
        res = evaluate_policy(name, score, labels, preds, correct, args.coverages, label_names)
        af_aurc, feasible_cov = aurc_under_af_floor(
            score, labels, preds, correct, af_idx, args.af_recall_floor
        )
        res["af_floor_aurc"] = af_aurc
        res["af_floor_feasible_cov"] = feasible_cov
        results[name] = res
        summary_rows.append({
            "policy": name,
            "aurc": res["aurc"],
            "sel_acc@0.5": res["sel_acc"]["0.5"],
            "sel_acc@0.7": res["sel_acc"]["0.7"],
            "af_recall@0.5": res["per_class"]["0.5"]["AF"]["recall_overall"],
            "af_recall@0.7": res["per_class"]["0.7"]["AF"]["recall_overall"],
            "af_floor_aurc": af_aurc,
            "af_floor_feasible_cov": feasible_cov,
        })

    # ---- Console summary ----
    print(f"\n=== Policy comparison (w={args.w}, AF-recall floor={args.af_recall_floor}) ===")
    print(f"  {'policy':<14}  {'AURC':>7}  {'sel@0.5':>8}  {'sel@0.7':>8}  "
          f"{'AF@0.5':>7}  {'AF@0.7':>7}  {'floor AURC':>11}  {'feasible cov':>13}")
    for row in summary_rows:
        print(
            f"  {row['policy']:<14}  "
            f"{row['aurc']:>7.4f}  "
            f"{row['sel_acc@0.5']:>8.4f}  {row['sel_acc@0.7']:>8.4f}  "
            f"{row['af_recall@0.5']:>7.3f}  {row['af_recall@0.7']:>7.3f}  "
            f"{row['af_floor_aurc']:>11.4f}  {row['af_floor_feasible_cov']:>13.3f}"
        )

    # ---- Per-class AF recall side-by-side (the key clinical view) ----
    print(f"\n=== AF recall at each coverage ===")
    print(f"  {'coverage':>8}  " + "  ".join(f"{n:>14}" for n in POLICIES))
    for c in args.coverages:
        cells = []
        for name in POLICIES:
            af = results[name]["per_class"][f"{c}"]["AF"]
            cells.append(f"{af['recall_overall']:.3f} ({af['n_kept']:>3d}/{af['n_total']})")
        print(f"  {c:>8.2f}  " + "  ".join(f"{x:>14}" for x in cells))

    # ---- Persist ----
    out_dir = args.predictions_csv.parent / f"class_conditional_sqi{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"summary_{args.signal_quality}_w{args.w:.2f}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / f"full_{args.signal_quality}_w{args.w:.2f}.json").open("w") as f:
        json.dump({
            "w": args.w,
            "signal_quality": args.signal_quality,
            "af_recall_floor": args.af_recall_floor,
            "policies": results,
        }, f, indent=2)
    print(f"\nWrote {out_dir}/summary_{args.signal_quality}_w{args.w:.2f}.csv")


if __name__ == "__main__":
    main()
