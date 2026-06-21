"""Continuous-w LW-CCSD optimization via Nelder-Mead with constraint penalty.

The grid version (optimize_class_conditional_weights.py) searches 11**3=1331
points and produces a slightly non-monotone test-side Pareto curve due to
grid discreteness. This script smooths it by Nelder-Mead on continuous
w in [0, 1]^3, with a soft penalty for constraint violation, and replicates
the floor-sweep across the Pareto frontier.

Usage:
    python scripts/continuous_lw_ccsd.py \\
        --config configs/synth_rppg_cinc.yaml \\
        --val-predictions runs/synth_rppg_cinc/eval_val_deterministic/predictions.csv \\
        --test-predictions runs/synth_rppg_cinc/eval_conformal/predictions.csv \\
        --run-dir runs/synth_rppg_cinc \\
        --af-recall-floor 0.55 --constraint-coverage 0.50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_class_conditional_weights import (  # noqa: E402
    _build_signals_split,
    _compute_quality,
    _load_predictions_csv,
    _rank_normalize,
    aurc,
    class_conditional_score,
    clopper_pearson_lower,
    per_class_recall_with_counts,
)
from rppg_sa.data.mitbih import CLASS_TO_INDEX
from rppg_sa.utils.config import load_config


def make_objective(
    conf_val: np.ndarray, sq_val: np.ndarray, preds_val: np.ndarray,
    labels_val: np.ndarray, num_classes: int,
    constraint_coverage: float, floors: np.ndarray,
    penalty_lambda: float = 100.0,
    conformal_alpha: float | None = None,
):
    correct_val = (preds_val == labels_val).astype(np.float64)

    def obj(w: np.ndarray) -> float:
        w_clipped = np.clip(w, 0.0, 1.0)
        score = class_conditional_score(conf_val, sq_val, preds_val, w_clipped)
        a = aurc(score, correct_val)
        rec, n_correct, n_total = per_class_recall_with_counts(
            score, labels_val, preds_val, constraint_coverage, num_classes
        )
        if conformal_alpha is not None:
            recall_metric = np.array([
                clopper_pearson_lower(int(n_correct[c]), int(n_total[c]), conformal_alpha)
                for c in range(num_classes)
            ])
        else:
            recall_metric = rec
        # Soft penalty: positive when constraint violated.
        viol = np.maximum(floors - recall_metric, 0.0)
        penalty = penalty_lambda * float((viol ** 2).sum())
        # Also penalize w outside [0,1] hypercube to keep Nelder-Mead in range.
        box_penalty = penalty_lambda * float(((np.maximum(w - 1.0, 0.0)) ** 2).sum())
        box_penalty += penalty_lambda * float(((np.maximum(-w, 0.0)) ** 2).sum())
        return a + penalty + box_penalty

    return obj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--val-predictions", type=Path, required=True)
    p.add_argument("--test-predictions", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--signal-quality", default="snr_db", choices=["snr_db", "template_sqi"])
    p.add_argument("--af-recall-floor", type=float, default=0.0)
    p.add_argument("--constraint-coverage", type=float, default=0.50)
    p.add_argument("--conformal-alpha", type=float, default=None)
    p.add_argument("--n-restarts", type=int, default=8,
                   help="Number of Nelder-Mead restarts from random initial w.")
    p.add_argument("--penalty-lambda", type=float, default=100.0)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    cfg = load_config(args.config)
    num_classes = int(cfg["classifier"]["num_classes"])
    af_idx = CLASS_TO_INDEX["AF"]

    # Load val.
    sig_val, fs, label_names = _build_signals_split(cfg, args.run_dir, "val")
    labels_val, preds_val, conf_val = _load_predictions_csv(args.val_predictions)
    sq_val = _compute_quality(sig_val, fs=fs, key=args.signal_quality)

    floors = np.zeros(num_classes, dtype=np.float64)
    floors[af_idx] = args.af_recall_floor
    print(f"floors @ cov={args.constraint_coverage}: {floors.tolist()}")
    print(f"conformal_alpha: {args.conformal_alpha}")
    print(f"restarts: {args.n_restarts}")

    obj = make_objective(
        conf_val, sq_val, preds_val, labels_val, num_classes,
        args.constraint_coverage, floors,
        penalty_lambda=args.penalty_lambda,
        conformal_alpha=args.conformal_alpha,
    )

    rng = np.random.default_rng(0)
    best = None
    history = []
    for restart in range(args.n_restarts):
        if restart == 0:
            x0 = np.array([0.3, 0.4, 0.4], dtype=np.float64)  # warm-start at grid optimum
        else:
            x0 = rng.uniform(0.0, 1.0, size=num_classes)
        res = minimize(
            obj, x0, method="Nelder-Mead",
            options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 500},
        )
        w = np.clip(res.x, 0.0, 1.0)
        # Re-evaluate clean (no penalty) val AURC + recall at the clipped w.
        score = class_conditional_score(conf_val, sq_val, preds_val, w)
        a_val = aurc(score, (preds_val == labels_val).astype(np.float64))
        rec, n_correct, n_total = per_class_recall_with_counts(
            score, labels_val, preds_val, args.constraint_coverage, num_classes
        )
        if args.conformal_alpha is not None:
            lcb = np.array([
                clopper_pearson_lower(int(n_correct[c]), int(n_total[c]), args.conformal_alpha)
                for c in range(num_classes)
            ])
            feasible = bool(np.all(lcb >= floors))
            af_metric = float(lcb[af_idx])
        else:
            feasible = bool(np.all(rec >= floors))
            af_metric = float(rec[af_idx])
        history.append({
            "restart": restart, "x0": x0.tolist(), "w": w.tolist(),
            "val_aurc": float(a_val), "val_af_metric": af_metric,
            "feasible": feasible,
        })
        if feasible and (best is None or a_val < best["val_aurc"]):
            best = {
                "w": w.tolist(), "val_aurc": float(a_val),
                "val_af_metric": af_metric,
            }

    if best is None:
        print("\nNo feasible w from any restart. Lower floor or increase restarts.")
        return

    w_star = np.array(best["w"], dtype=np.float64)
    print(f"\n*** Continuous w* = {dict(zip(label_names, [round(v, 4) for v in best['w']]))}")
    print(f"    val AURC: {best['val_aurc']:.4f}   val AF metric: {best['val_af_metric']:.4f}")

    # Apply to test.
    sig_test, fs_test, _ = _build_signals_split(cfg, args.run_dir, "test")
    labels_test, preds_test, conf_test = _load_predictions_csv(args.test_predictions)
    sq_test = _compute_quality(sig_test, fs=fs_test, key=args.signal_quality)
    correct_test = (preds_test == labels_test).astype(np.float64)
    score_uq = _rank_normalize(conf_test)
    score_lw = class_conditional_score(conf_test, sq_test, preds_test, w_star)
    a_uq = aurc(score_uq, correct_test)
    a_lw = aurc(score_lw, correct_test)
    rec_uq, _, _ = per_class_recall_with_counts(
        score_uq, labels_test, preds_test, args.constraint_coverage, num_classes
    )
    rec_lw, _, _ = per_class_recall_with_counts(
        score_lw, labels_test, preds_test, args.constraint_coverage, num_classes
    )
    delta = a_uq - a_lw
    print(f"\n  test AURC:  {a_uq:.4f}  -> {a_lw:.4f}  (Delta={delta:+.4f}, "
          f"{100*delta/a_uq:+.2f} %)")
    print(f"  AF recall @ cov={args.constraint_coverage}: "
          f"{rec_uq[af_idx]:.3f}  -> {rec_lw[af_idx]:.3f}  "
          f"(Delta={rec_lw[af_idx]-rec_uq[af_idx]:+.3f})")

    out_dir = args.test_predictions.parent / f"lw_ccsd_continuous{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "continuous_weights.json").open("w") as f:
        json.dump({
            "w_star": dict(zip(label_names, best["w"])),
            "val_aurc": best["val_aurc"],
            "test_aurc_uq_only": float(a_uq),
            "test_aurc_lw_ccsd": float(a_lw),
            "test_af_recall_uq": float(rec_uq[af_idx]),
            "test_af_recall_lw": float(rec_lw[af_idx]),
            "constraint_coverage": args.constraint_coverage,
            "af_recall_floor": args.af_recall_floor,
            "conformal_alpha": args.conformal_alpha,
            "history": history,
        }, f, indent=2)
    print(f"  -> {out_dir}/continuous_weights.json")


if __name__ == "__main__":
    main()
