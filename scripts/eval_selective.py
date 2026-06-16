"""Real selective-prediction evaluation script.

Loads a trained checkpoint, runs MC-Dropout inference on the held-out test set,
computes the full suite of selective-prediction metrics, and writes:
    - results.json: scalar metrics
    - risk_coverage.csv: per-coverage risk for plotting
    - reliability.csv: per-bin calibration data

Usage:
    python scripts/eval_selective.py \\
        --config configs/mitbih_baseline.yaml \\
        --checkpoint runs/mitbih_baseline/best.pt \\
        --uq mc_dropout --mc-samples 30
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rppg_sa.models.cnn1d_transformer import CNN1DTransformer
from rppg_sa.selective.metrics import (
    brier_score,
    expected_calibration_error,
    predictive_entropy,
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)
from rppg_sa.uncertainty.mc_dropout import mc_dropout_predict
from rppg_sa.utils.config import load_config
from rppg_sa.utils.seed import set_seed


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _build_test_loader(cfg: dict[str, Any], run_dir: Path, batch_size: int = 64):
    source = cfg["data"]["source"]
    if source != "mitbih":
        raise ValueError(f"Unsupported data source for eval: {source}")
    from rppg_sa.data.mitbih_torch import MITBIHSegmentDataset

    ds = MITBIHSegmentDataset(
        root=cfg["data"]["root"],
        target_fs=float(cfg["data"]["target_fs"]),
        window_seconds=float(cfg["data"]["window_seconds"]),
    )
    splits_file = run_dir / "splits.json"
    if splits_file.exists():
        with splits_file.open() as f:
            test_idx = json.load(f)["test_idx"]
    else:
        # Fallback: derive from config.
        from rppg_sa.data.mitbih_torch import subject_disjoint_split

        train_ids = [str(r) for r in cfg["data"]["splits"]["train"]]
        val_ids = [str(r) for r in cfg["data"]["splits"]["val"]]
        test_ids = [str(r) for r in cfg["data"]["splits"]["test"]]
        _, _, test_idx = subject_disjoint_split(ds, train_ids, val_ids, test_ids)
    test_ds = Subset(ds, test_idx)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, ds.LABEL_NAMES


def _reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> list[dict]:
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": int(in_bin.sum()),
                "mean_confidence": float(confs[in_bin].mean()) if in_bin.any() else float("nan"),
                "accuracy": float(correct[in_bin].mean()) if in_bin.any() else float("nan"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective prediction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--uq", choices=["mc_dropout"], default="mc_dropout")
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument(
        "--coverages",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))
    device = _resolve_device(cfg["experiment"]["device"])
    print(f"Device: {device}")

    # ---------- Model ----------
    mcfg = cfg["classifier"]
    model = CNN1DTransformer(
        in_channels=int(mcfg["in_channels"]),
        num_classes=int(mcfg["num_classes"]),
        conv_channels=tuple(mcfg["conv_channels"]),
        transformer_layers=int(mcfg["transformer_layers"]),
        transformer_heads=int(mcfg["transformer_heads"]),
        dropout=float(mcfg["dropout"]),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---------- Data ----------
    run_dir = args.checkpoint.parent
    loader, label_names = _build_test_loader(cfg, run_dir)
    num_classes = int(mcfg["num_classes"])

    # ---------- Predict with MC Dropout ----------
    all_probs: list[np.ndarray] = []
    all_entropy: list[np.ndarray] = []
    all_mutual_info: list[np.ndarray] = []
    all_labels: list[int] = []
    for batch in loader:
        x = batch["signal"].to(device)
        probs, entropy, mi = mc_dropout_predict(model, x, num_samples=args.mc_samples)
        all_probs.append(probs)
        all_entropy.append(entropy)
        all_mutual_info.append(mi)
        all_labels.extend(batch["label"].tolist())

    probs = np.concatenate(all_probs, axis=0)
    entropy = np.concatenate(all_entropy, axis=0)
    mutual_info = np.concatenate(all_mutual_info, axis=0)
    labels = np.array(all_labels, dtype=np.int64)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)

    # ---------- Metrics ----------
    accuracy = float(correct.mean())
    ece = expected_calibration_error(probs, labels, n_bins=15)
    bs = brier_score(probs, labels, num_classes=num_classes)
    # Confidence = -entropy (so that high confidence sorts first).
    confidences = -entropy
    rc = risk_coverage_curve(confidences, correct)
    selective_acc = {
        f"sel_acc@{c}": selective_accuracy_at_coverage(confidences, correct, c)
        for c in args.coverages
    }
    # Per-class breakdown.
    per_class: dict[str, dict[str, float]] = {}
    for c in range(num_classes):
        m = labels == c
        if not m.any():
            continue
        per_class[label_names[c]] = {
            "n": int(m.sum()),
            "accuracy": float(correct[m].mean()),
            "mean_entropy": float(entropy[m].mean()),
        }

    results = {
        "uq_method": args.uq,
        "mc_samples": args.mc_samples,
        "test_n": int(len(labels)),
        "accuracy": accuracy,
        "ece": ece,
        "brier_score": bs,
        "aurc": rc.aurc,
        "mean_predictive_entropy": float(entropy.mean()),
        "mean_mutual_info": float(mutual_info.mean()),
        **selective_acc,
        "per_class": per_class,
    }

    out_dir = run_dir / f"eval_{args.uq}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # Risk-coverage CSV.
    with (out_dir / "risk_coverage.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coverage", "risk"])
        for cov, r in zip(rc.coverage, rc.risk):
            w.writerow([f"{cov:.6f}", f"{r:.6f}"])

    # Reliability CSV.
    with (out_dir / "reliability.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bin_lo", "bin_hi", "n", "mean_confidence", "accuracy"])
        w.writeheader()
        for row in _reliability_table(probs, labels):
            w.writerow(row)

    # Per-sample diagnostics (label, pred, probs, entropy, MI).
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label", "pred", "entropy", "mutual_info"] + [f"p_{n}" for n in label_names])
        for i, (lbl, prd, h, mi, p) in enumerate(zip(labels, preds, entropy, mutual_info, probs)):
            w.writerow([i, int(lbl), int(prd), f"{h:.6f}", f"{mi:.6f}", *[f"{v:.6f}" for v in p]])

    print(f"Wrote artefacts to {out_dir}")


if __name__ == "__main__":
    main()
