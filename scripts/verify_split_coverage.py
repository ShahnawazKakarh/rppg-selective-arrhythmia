"""Per-split class-coverage verification.

Catches the v0 failure mode: a naive auto-split that drops an entire class
from one of train/val/test, which silently produces meaningless metrics
downstream.

The script rebuilds the dataset and computes the train/val/test indices using
the same code path as scripts/train_classifier.py, then asserts:
  - every class is present in every split
  - per-class count in each split is >= a configurable minimum (default 1)
  - per-class fraction in each split is within a configurable tolerance of
    the global class distribution (default: warn-only, no hard fail)

Usage:
    python scripts/verify_split_coverage.py --config configs/synth_rppg_cinc.yaml
    python scripts/verify_split_coverage.py --config configs/mitbih_baseline.yaml --min-per-class 5

Exit codes:
    0 = all checks pass
    1 = at least one class missing from at least one split
    2 = config or dataset load failure
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the dataset builder from the trainer so split logic stays in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_classifier import _build_dataset  # noqa: E402

from rppg_sa.utils.config import load_config  # noqa: E402


def _class_counts(labels: list[int], num_classes: int) -> np.ndarray:
    counts = Counter(labels)
    return np.array([counts.get(c, 0) for c in range(num_classes)], dtype=np.int64)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data-seed", type=int, default=None,
                   help="Override cfg.experiment.data_seed (otherwise uses config value).")
    p.add_argument("--min-per-class", type=int, default=1,
                   help="Hard minimum count per class per split (default: 1).")
    p.add_argument("--distribution-tolerance", type=float, default=0.10,
                   help="Maximum allowed absolute deviation in per-class fraction between "
                        "any split and the global distribution. Warn-only by default.")
    p.add_argument("--strict-distribution", action="store_true",
                   help="If set, distribution deviations beyond tolerance fail the check.")
    args = p.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"ERROR: failed to load config {args.config}: {e}", file=sys.stderr)
        return 2

    # Wire data_seed / model_seed into cfg the way the trainer does, so the
    # split decision matches what training would see.
    legacy_seed = int(cfg["experiment"].get("seed", 0))
    data_seed = args.data_seed
    if data_seed is None:
        data_seed = int(cfg["experiment"].get("data_seed", legacy_seed))
    cfg["experiment"]["data_seed"] = int(data_seed)
    cfg["experiment"]["model_seed"] = int(cfg["experiment"].get("model_seed", legacy_seed))
    cfg["experiment"]["seed"] = int(cfg["experiment"]["model_seed"])

    try:
        full_ds, train_idx, val_idx, test_idx, label_names = _build_dataset(cfg)
    except Exception as e:
        print(f"ERROR: failed to build dataset: {e}", file=sys.stderr)
        return 2

    num_classes = len(label_names)
    train_labels = [full_ds.labels[i] for i in train_idx]
    val_labels = [full_ds.labels[i] for i in val_idx]
    test_labels = [full_ds.labels[i] for i in test_idx]
    full_labels = [full_ds.labels[i] for i in range(len(full_ds))]

    train_counts = _class_counts(train_labels, num_classes)
    val_counts = _class_counts(val_labels, num_classes)
    test_counts = _class_counts(test_labels, num_classes)
    full_counts = _class_counts(full_labels, num_classes)

    splits: list[tuple[str, np.ndarray]] = [
        ("train", train_counts),
        ("val", val_counts),
        ("test", test_counts),
    ]

    # ---- Header ----
    print(f"Config:     {args.config}")
    print(f"Source:     {cfg['data']['source']}")
    print(f"data_seed:  {cfg['experiment']['data_seed']}")
    print(f"Dataset:    {len(full_ds)} segments  |  classes: {label_names}")
    print()

    # ---- Per-class counts table ----
    header = f"  {'class':<10}  {'train':>10}  {'val':>10}  {'test':>10}  {'total':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c, name in enumerate(label_names):
        print(f"  {name:<10}  {train_counts[c]:>10}  {val_counts[c]:>10}  "
              f"{test_counts[c]:>10}  {full_counts[c]:>10}")
    print(f"  {'TOTAL':<10}  {train_counts.sum():>10}  {val_counts.sum():>10}  "
          f"{test_counts.sum():>10}  {full_counts.sum():>10}")
    print()

    # ---- Per-class fraction table ----
    if full_counts.sum() > 0:
        global_frac = full_counts / float(full_counts.sum())
    else:
        global_frac = np.zeros(num_classes)
    print(f"  {'class':<10}  {'train%':>10}  {'val%':>10}  {'test%':>10}  {'global%':>10}")
    print("  " + "-" * (len(header) - 2))
    train_frac = train_counts / max(int(train_counts.sum()), 1)
    val_frac = val_counts / max(int(val_counts.sum()), 1)
    test_frac = test_counts / max(int(test_counts.sum()), 1)
    for c, name in enumerate(label_names):
        print(f"  {name:<10}  {100*train_frac[c]:>9.2f}%  {100*val_frac[c]:>9.2f}%  "
              f"{100*test_frac[c]:>9.2f}%  {100*global_frac[c]:>9.2f}%")
    print()

    # ---- Hard check: minimum per-class count per split ----
    failures = []
    for name, counts in splits:
        for c, n_c in enumerate(counts):
            if n_c < args.min_per_class:
                failures.append(
                    f"FAIL  split={name:<5} class={label_names[c]:<6} "
                    f"count={n_c} (< min_per_class={args.min_per_class})"
                )

    # ---- Soft / strict check: distribution deviation ----
    warnings = []
    for name, counts in splits:
        if counts.sum() == 0:
            continue
        frac = counts / float(counts.sum())
        for c in range(num_classes):
            dev = abs(frac[c] - global_frac[c])
            if dev > args.distribution_tolerance:
                msg = (f"distribution deviation: split={name:<5} class={label_names[c]:<6} "
                       f"{100*frac[c]:.2f}% vs global {100*global_frac[c]:.2f}% "
                       f"(|Δ|={100*dev:.2f}%, tol={100*args.distribution_tolerance:.2f}%)")
                if args.strict_distribution:
                    failures.append("FAIL  " + msg)
                else:
                    warnings.append("WARN  " + msg)

    # ---- Report ----
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  {f}")
        print()
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")
        print()

    if failures:
        print(f"Result: FAIL  ({len(failures)} failure(s), {len(warnings)} warning(s))")
        return 1
    print(f"Result: PASS  ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
