"""Learn class-conditional SQI weights (LW-CCSD).

The methodological contribution: instead of hand-tuning a single quality
weight w (the `af_immune` rule's binary on/off), *learn* a per-predicted-
class quality weight vector w = (w_NSR, w_AF, w_Other) on validation data
by constrained grid search:

    minimize_w  AURC(val; w)
    subject to  per-class recall on val at coverage c0 >= floor_c   for c in classes
                w_c in [0, 1]

The optimum w* is then applied unchanged to the test set. This is a
post-hoc, model-agnostic deferral policy with formal per-class safety
guarantees on the validation distribution.

For 3 classes and an 11-point grid this is 11**3 = 1331 candidates —
fully tractable in seconds with numpy.

Outputs:
    learned_weights.json   per-class w*, val/test AURC, val/test per-class
                           recall under both UQ-only and LW-CCSD policies.
    sweep.csv              every w combination tried with val AURC + recalls.

Usage:
    python scripts/optimize_class_conditional_weights.py \\
        --config configs/synth_rppg_cinc.yaml \\
        --val-predictions runs/synth_rppg_cinc/eval_val_deterministic/predictions.csv \\
        --test-predictions runs/synth_rppg_cinc/eval_conformal/predictions.csv \\
        --run-dir runs/synth_rppg_cinc \\
        --signal-quality snr_db \\
        --af-recall-floor 0.70
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta as _beta_dist

from rppg_sa.data.mitbih import CLASS_TO_INDEX
from rppg_sa.extractors.signal_quality import summarize_quality
from rppg_sa.selective.metrics import (
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)
from rppg_sa.utils.config import load_config


DEFAULT_GRID = np.arange(0.0, 1.001, 0.10).round(2).tolist()   # 11 points
DEFAULT_COVERAGES = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------
def _build_signals_split(cfg: dict[str, Any], run_dir: Path, split: str):
    """Return list[np.ndarray] of pulse waveforms for split in {val, test}."""
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
        idx = sp[f"{split}_idx"]
    else:
        train_ids = [str(r) for r in cfg["data"]["splits"]["train"]]
        val_ids = [str(r) for r in cfg["data"]["splits"]["val"]]
        test_ids = [str(r) for r in cfg["data"]["splits"]["test"]]
        tr, va, te = subject_disjoint_split(ds, train_ids, val_ids, test_ids)
        idx = {"train": tr, "val": va, "test": te}[split]
    return [ds.signals[i] for i in idx], float(cfg["data"]["target_fs"]), ds.LABEL_NAMES


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
# Scoring + metrics
# ----------------------------------------------------------------------
def _rank_normalize(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order.astype(np.float64) / max(len(x) - 1, 1)


def class_conditional_score(
    conf: np.ndarray,
    sq: np.ndarray,
    preds: np.ndarray,
    w_per_class: np.ndarray,
) -> np.ndarray:
    """combined[i] = (1 - w[pred_i]) * conf_rank[i] + w[pred_i] * sq_rank[i].

    w_per_class is indexed by predicted class label.
    """
    conf_r = _rank_normalize(conf)
    sq_r = _rank_normalize(sq)
    w_i = w_per_class[preds]
    return (1.0 - w_i) * conf_r + w_i * sq_r


def per_class_recall_at_coverage(
    score: np.ndarray, labels: np.ndarray, preds: np.ndarray,
    coverage: float, num_classes: int,
) -> np.ndarray:
    """recall[c] = (# correctly predicted class c kept) / (# class c in total)."""
    rec, _, _ = per_class_recall_with_counts(
        score, labels, preds, coverage, num_classes
    )
    return rec


def per_class_recall_with_counts(
    score: np.ndarray, labels: np.ndarray, preds: np.ndarray,
    coverage: float, num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as per_class_recall_at_coverage but also returns (n_correct_kept, n_total) per class.

    Counts are needed for the Clopper-Pearson lower bound used by the
    conformal LW-CCSD variant.
    """
    n = len(score)
    k = max(int(round(coverage * n)), 1)
    if k >= n:
        kept = np.arange(n)
    else:
        kept = np.argpartition(score, -k)[-k:]
    kept_mask = np.zeros(n, dtype=bool)
    kept_mask[kept] = True

    recall = np.zeros(num_classes, dtype=np.float64)
    n_correct_arr = np.zeros(num_classes, dtype=np.int64)
    n_total_arr = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        m_class = labels == c
        n_total = int(m_class.sum())
        n_correct_kept = int(((preds == labels) & m_class & kept_mask).sum())
        n_total_arr[c] = n_total
        n_correct_arr[c] = n_correct_kept
        recall[c] = n_correct_kept / max(n_total, 1)
    return recall, n_correct_arr, n_total_arr


def clopper_pearson_lower(n_correct: int, n_total: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower bound on a binomial proportion.

    Returns LCB such that P(true p >= LCB) >= 1 - alpha (under exchangeability).
    """
    if n_total == 0:
        return 0.0
    if n_correct == 0:
        return 0.0
    return float(_beta_dist.ppf(alpha, n_correct, n_total - n_correct + 1))


def aurc(score: np.ndarray, correct: np.ndarray) -> float:
    return float(risk_coverage_curve(score, correct).aurc)


# ----------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------
def optimize_weights(
    conf_val: np.ndarray, sq_val: np.ndarray, preds_val: np.ndarray,
    labels_val: np.ndarray, num_classes: int,
    grid: list[float], constraint_coverage: float,
    recall_floors: np.ndarray,
    conformal_alpha: float | None = None,
) -> dict[str, Any]:
    """Constrained grid search over per-class quality weights.

    Among all w in grid^num_classes satisfying val per-class recall >= floor
    at constraint_coverage, return the one that minimizes val AURC.

    If conformal_alpha is given (e.g. 0.10), the floor is enforced on the
    Clopper-Pearson one-sided lower confidence bound of val recall rather
    than the point estimate, yielding P(true recall >= floor) >= 1 - alpha
    under exchangeability.
    """
    correct_val = (preds_val == labels_val).astype(np.float64)
    grid_arr = np.array(grid, dtype=np.float64)

    rows = []
    best = None
    for combo in product(grid_arr, repeat=num_classes):
        w = np.array(combo, dtype=np.float64)
        score = class_conditional_score(conf_val, sq_val, preds_val, w)
        a = aurc(score, correct_val)
        rec, n_correct, n_total = per_class_recall_with_counts(
            score, labels_val, preds_val, constraint_coverage, num_classes
        )
        if conformal_alpha is not None:
            lcb = np.array([
                clopper_pearson_lower(int(n_correct[c]), int(n_total[c]), conformal_alpha)
                for c in range(num_classes)
            ])
            feasible = bool(np.all(lcb >= recall_floors))
            row_recall_field = lcb.tolist()
        else:
            feasible = bool(np.all(rec >= recall_floors))
            row_recall_field = rec.tolist()
        rows.append({
            "w": w.tolist(),
            "val_aurc": a,
            "val_recall_per_class": rec.tolist(),
            "val_recall_lcb_per_class": (
                row_recall_field if conformal_alpha is not None else None
            ),
            "feasible": feasible,
        })
        if feasible and (best is None or a < best["val_aurc"]):
            best = {
                "w": w.tolist(),
                "val_aurc": a,
                "val_recall_per_class": rec.tolist(),
                "val_recall_lcb_per_class": (
                    row_recall_field if conformal_alpha is not None else None
                ),
            }
    return {"all_candidates": rows, "best": best}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--val-predictions", type=Path, required=True)
    p.add_argument("--test-predictions", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--signal-quality", default="snr_db", choices=["snr_db", "template_sqi"])
    p.add_argument("--af-recall-floor", type=float, default=0.70)
    p.add_argument("--other-recall-floor", type=float, default=0.0,
                   help="Floor for Other class; default 0 (no constraint).")
    p.add_argument("--nsr-recall-floor", type=float, default=0.0)
    p.add_argument("--constraint-coverage", type=float, default=0.70,
                   help="Coverage at which the recall floor must hold (on val).")
    p.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    p.add_argument("--coverages", type=float, nargs="+", default=DEFAULT_COVERAGES)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--conformal-alpha", type=float, default=None,
                   help="If set (e.g. 0.10), enforce per-class recall floor on the "
                        "Clopper-Pearson one-sided lower confidence bound of val recall, "
                        "yielding P(true recall >= floor) >= 1 - alpha under exchangeability.")
    args = p.parse_args()

    cfg = load_config(args.config)
    num_classes = int(cfg["classifier"]["num_classes"])
    af_idx = CLASS_TO_INDEX["AF"]
    nsr_idx = CLASS_TO_INDEX["NSR"]
    other_idx = CLASS_TO_INDEX["Other"]

    # ---- Val side ----
    val_signals, fs, label_names = _build_signals_split(cfg, args.run_dir, "val")
    labels_val, preds_val, conf_val = _load_predictions_csv(args.val_predictions)
    assert len(labels_val) == len(val_signals), (
        f"val length mismatch: predictions={len(labels_val)} vs signals={len(val_signals)}"
    )
    print(f"val: n={len(labels_val)}  classes={label_names}  fs={fs}")
    print(f"Computing {args.signal_quality} on val ...")
    sq_val = _compute_quality(val_signals, fs=fs, key=args.signal_quality)

    floors = np.zeros(num_classes, dtype=np.float64)
    floors[nsr_idx] = args.nsr_recall_floor
    floors[af_idx] = args.af_recall_floor
    floors[other_idx] = args.other_recall_floor
    print(f"Recall floors (constraint_coverage={args.constraint_coverage}): {floors.tolist()}")
    print(f"Grid: {args.grid}  ({len(args.grid) ** num_classes} candidates)")

    # Sanity: print UQ-only val per-class recall as the achievable baseline.
    correct_val = (preds_val == labels_val).astype(np.float64)
    score_uq_val = _rank_normalize(conf_val)
    rec_uq = per_class_recall_at_coverage(
        score_uq_val, labels_val, preds_val, args.constraint_coverage, num_classes
    )
    print(f"UQ-only val per-class recall @ cov={args.constraint_coverage}: "
          f"{dict(zip(label_names, rec_uq.tolist()))}  "
          f"(val AURC = {aurc(score_uq_val, correct_val):.4f})")

    opt = optimize_weights(
        conf_val, sq_val, preds_val, labels_val, num_classes,
        args.grid, args.constraint_coverage, floors,
        conformal_alpha=args.conformal_alpha,
    )
    if opt["best"] is None:
        if args.conformal_alpha is not None:
            print(f"\nNo feasible weight combination on val under conformal alpha={args.conformal_alpha}. "
                  f"Lower recall floors, increase alpha, or expand grid.")
        else:
            print("\nNo feasible weight combination on val. Lower recall floors or expand grid.")
        return
    w_star = np.array(opt["best"]["w"], dtype=np.float64)
    print(f"\n*** Optimal weights w* = {dict(zip(label_names, w_star.tolist()))}")
    print(f"    val AURC at w*: {opt['best']['val_aurc']:.4f}")
    print(f"    val per-class recall @ cov={args.constraint_coverage}: "
          f"{dict(zip(label_names, opt['best']['val_recall_per_class']))}")

    # ---- Test side ----
    test_signals, _, _ = _build_signals_split(cfg, args.run_dir, "test")
    labels_te, preds_te, conf_te = _load_predictions_csv(args.test_predictions)
    assert len(labels_te) == len(test_signals), (
        f"test length mismatch: predictions={len(labels_te)} vs signals={len(test_signals)}"
    )
    print(f"\ntest: n={len(labels_te)}")
    print(f"Computing {args.signal_quality} on test ...")
    sq_te = _compute_quality(test_signals, fs=fs, key=args.signal_quality)

    correct_te = (preds_te == labels_te).astype(np.float64)
    score_uq = _rank_normalize(conf_te)
    score_lw = class_conditional_score(conf_te, sq_te, preds_te, w_star)

    summary = {"policies": {}}
    for name, score in [("uq_only", score_uq), ("lw_ccsd", score_lw)]:
        a = aurc(score, correct_te)
        sel = {f"sel_acc@{c}": selective_accuracy_at_coverage(score, correct_te, c)
               for c in args.coverages}
        per_class = {}
        for c in args.coverages:
            rec = per_class_recall_at_coverage(score, labels_te, preds_te, c, num_classes)
            per_class[f"{c}"] = {label_names[k]: float(rec[k]) for k in range(num_classes)}
        summary["policies"][name] = {
            "test_aurc": a,
            "test_sel_acc": sel,
            "test_per_class_recall": per_class,
        }
        print(f"\n[{name}] test AURC = {a:.4f}")
        for c in args.coverages:
            rec_str = "  ".join(
                f"{label_names[k]}={per_class[f'{c}'][label_names[k]]:.3f}"
                for k in range(num_classes)
            )
            print(f"  cov={c:.2f}  sel_acc={sel[f'sel_acc@{c}']:.4f}  {rec_str}")

    # ---- Side-by-side delta ----
    uq = summary["policies"]["uq_only"]
    lw = summary["policies"]["lw_ccsd"]
    delta_aurc = uq["test_aurc"] - lw["test_aurc"]
    print(f"\n=== test: LW-CCSD vs UQ-only ===")
    print(f"  test AURC:  {uq['test_aurc']:.4f}  -> {lw['test_aurc']:.4f}  (Δ={delta_aurc:+.4f}, "
          f"{100*delta_aurc/uq['test_aurc']:+.2f} %)")
    for c in args.coverages:
        u_af = uq["test_per_class_recall"][f"{c}"]["AF"]
        l_af = lw["test_per_class_recall"][f"{c}"]["AF"]
        print(f"  AF recall @ cov={c:.2f}:  {u_af:.3f}  -> {l_af:.3f}  (Δ={l_af - u_af:+.3f})")

    # ---- Persist ----
    out_dir = args.test_predictions.parent / f"lw_ccsd{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    learned = {
        "signal_quality": args.signal_quality,
        "grid": args.grid,
        "af_recall_floor": args.af_recall_floor,
        "constraint_coverage": args.constraint_coverage,
        "conformal_alpha": args.conformal_alpha,
        "label_names": label_names,
        "w_star": dict(zip(label_names, w_star.tolist())),
        "val_best": opt["best"],
        "test": summary,
    }
    with (out_dir / "learned_weights.json").open("w") as f:
        json.dump(learned, f, indent=2)
    with (out_dir / "sweep.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["w_" + n for n in label_names] + ["val_aurc", "feasible"]
                        + ["val_recall_" + n for n in label_names])
        for row in opt["all_candidates"]:
            writer.writerow([*row["w"], f"{row['val_aurc']:.6f}", int(row["feasible"]),
                             *[f"{r:.6f}" for r in row["val_recall_per_class"]]])
    print(f"\nWrote {out_dir}/learned_weights.json + sweep.csv")


if __name__ == "__main__":
    main()
