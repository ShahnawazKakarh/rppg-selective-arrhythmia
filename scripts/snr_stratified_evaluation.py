"""SNR-stratified evaluation of LW-CCSD vs UQ-only on the CinC test set.

Bins the test set by spectral SNR (cardiac band) into three regimes
(low / mid / high) and reports per-bin:
  - UQ-only AURC
  - LW-CCSD AURC (using a w* from the main run)
  - per-class recall @ cov=0.5 under both policies

Answers the question: does the LW-CCSD gain concentrate in a specific
signal-quality regime, or is it uniform? Important robustness check
for the paper.

Usage:
    python scripts/snr_stratified_evaluation.py \\
        --config configs/synth_rppg_cinc.yaml \\
        --test-predictions runs/synth_rppg_cinc/eval_conformal/predictions.csv \\
        --run-dir runs/synth_rppg_cinc \\
        --w-nsr 0.3 --w-af 0.4 --w-other 0.4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rppg_sa.data.mitbih import CLASS_TO_INDEX
from rppg_sa.extractors.signal_quality import summarize_quality
from rppg_sa.selective.metrics import risk_coverage_curve
from rppg_sa.utils.config import load_config

# Reuse helpers from the main optimizer to avoid divergence.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_class_conditional_weights import (  # noqa: E402
    _build_signals_split,
    _load_predictions_csv,
    _compute_quality,
    _rank_normalize,
    class_conditional_score,
    per_class_recall_at_coverage,
    aurc,
)


def stratify_by_tertile(sq: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Return (bin_idx, bin_edges) splitting sq into low/mid/high tertiles."""
    q33, q66 = np.percentile(sq, [33.33, 66.67])
    edges = [
        (float(sq.min()), float(q33)),
        (float(q33), float(q66)),
        (float(q66), float(sq.max())),
    ]
    bins = np.zeros(len(sq), dtype=np.int64)
    bins[sq >= q33] = 1
    bins[sq >= q66] = 2
    return bins, edges


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--test-predictions", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--signal-quality", default="snr_db", choices=["snr_db", "template_sqi"])
    p.add_argument("--w-nsr", type=float, required=True)
    p.add_argument("--w-af", type=float, required=True)
    p.add_argument("--w-other", type=float, required=True)
    p.add_argument("--coverage", type=float, default=0.50,
                   help="Coverage at which per-class recall is reported.")
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    cfg = load_config(args.config)
    num_classes = int(cfg["classifier"]["num_classes"])
    nsr_idx = CLASS_TO_INDEX["NSR"]
    af_idx = CLASS_TO_INDEX["AF"]
    other_idx = CLASS_TO_INDEX["Other"]

    w_per_class = np.zeros(num_classes, dtype=np.float64)
    w_per_class[nsr_idx] = args.w_nsr
    w_per_class[af_idx] = args.w_af
    w_per_class[other_idx] = args.w_other

    signals, fs, label_names = _build_signals_split(cfg, args.run_dir, "test")
    labels, preds, conf = _load_predictions_csv(args.test_predictions)
    assert len(signals) == len(labels)
    sq = _compute_quality(signals, fs=fs, key=args.signal_quality)
    print(f"test: n={len(labels)}  fs={fs}  classes={label_names}")
    print(f"using w* = {dict(zip(label_names, w_per_class.tolist()))}")

    # Global metrics for context.
    correct = (preds == labels).astype(np.float64)
    score_uq_global = _rank_normalize(conf)
    score_lw_global = class_conditional_score(conf, sq, preds, w_per_class)
    a_uq_global = aurc(score_uq_global, correct)
    a_lw_global = aurc(score_lw_global, correct)
    print(f"\nGlobal: UQ-only AURC = {a_uq_global:.4f}   LW-CCSD AURC = {a_lw_global:.4f}   "
          f"Delta = {(a_uq_global - a_lw_global):+.4f}  "
          f"({100*(a_uq_global - a_lw_global)/a_uq_global:+.2f}%)")

    # Stratify.
    bin_idx, edges = stratify_by_tertile(sq)
    bin_names = [
        f"low [{edges[0][0]:.1f}, {edges[0][1]:.1f}]\u202fdB",
        f"mid [{edges[1][0]:.1f}, {edges[1][1]:.1f}]\u202fdB",
        f"high [{edges[2][0]:.1f}, {edges[2][1]:.1f}]\u202fdB",
    ]

    out: dict[str, Any] = {
        "w_star": dict(zip(label_names, w_per_class.tolist())),
        "global": {
            "uq_only_aurc": a_uq_global,
            "lw_ccsd_aurc": a_lw_global,
            "delta": a_uq_global - a_lw_global,
        },
        "bins": [],
    }

    print(f"\nPer-tertile evaluation (signal_quality = {args.signal_quality}):")
    header = (f"{'Bin':<28} {'n':>6}  {'NSR':>4} {'AF':>4} {'Other':>6}  "
              f"{'UQ AURC':>9} {'LW AURC':>9}  {'Δ AURC':>9} {'Δ %':>7}  "
              f"{'UQ AF@.5':>9} {'LW AF@.5':>9}")
    print(header)
    print("-" * len(header))

    # IMPORTANT METHODOLOGICAL NOTE
    # ----------------------------
    # The LW-CCSD score combines two RANK-normalised features over the *test*
    # set. Rank-normalisation makes the score dependent on the population over
    # which the rank is computed. Two valid interpretations:
    #   (A) "Cohort policy": rank over the full test set, then slice. Mirrors
    #       deployment where the deferral threshold is set globally; the bin
    #       only changes which subset of the global ranking we evaluate.
    #   (B) "Local-cohort policy": rerank within the bin. Mirrors deployment
    #       where the system *also* knows the SNR regime at inference time.
    # We report (A) — the deployment-relevant case. Internal ranks within the
    # bin are still defined; what changes is whether the ranking incorporates
    # cross-bin signal.

    for b in range(3):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            print(f"{bin_names[b]:<28} {n_b:>6}  (empty)")
            continue

        # Cohort policy: slice the global score, then re-rank within bin to
        # recompute per-bin AURC (which is an *ordering*-based metric local
        # to the bin). The slicing preserves the global w* decisions; only
        # the AURC arithmetic is local.
        sub_correct = correct[mask]
        sub_labels = labels[mask]
        sub_preds = preds[mask]
        sub_score_uq = score_uq_global[mask]
        sub_score_lw = score_lw_global[mask]

        a_uq_b = aurc(sub_score_uq, sub_correct)
        a_lw_b = aurc(sub_score_lw, sub_correct)

        per_class_uq = per_class_recall_at_coverage(
            sub_score_uq, sub_labels, sub_preds, args.coverage, num_classes
        )
        per_class_lw = per_class_recall_at_coverage(
            sub_score_lw, sub_labels, sub_preds, args.coverage, num_classes
        )

        class_counts = [int((sub_labels == c).sum()) for c in range(num_classes)]
        delta = a_uq_b - a_lw_b
        delta_pct = 100 * delta / max(a_uq_b, 1e-9)

        print(f"{bin_names[b]:<28} {n_b:>6}  "
              f"{class_counts[nsr_idx]:>4} {class_counts[af_idx]:>4} {class_counts[other_idx]:>6}  "
              f"{a_uq_b:>9.4f} {a_lw_b:>9.4f}  {delta:>+9.4f} {delta_pct:>+6.2f}%  "
              f"{per_class_uq[af_idx]:>9.3f} {per_class_lw[af_idx]:>9.3f}")

        out["bins"].append({
            "name": bin_names[b],
            "edges_db": edges[b],
            "n": n_b,
            "class_counts": dict(zip(label_names, class_counts)),
            "uq_only": {
                "aurc": a_uq_b,
                "per_class_recall_at_coverage": dict(zip(label_names, per_class_uq.tolist())),
            },
            "lw_ccsd": {
                "aurc": a_lw_b,
                "per_class_recall_at_coverage": dict(zip(label_names, per_class_lw.tolist())),
            },
            "delta_aurc": delta,
            "delta_aurc_pct": delta_pct,
        })

    # ---- Persist ----
    out_dir = args.test_predictions.parent / f"snr_stratified{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "snr_stratified.json").open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_dir}/snr_stratified.json")


if __name__ == "__main__":
    main()
