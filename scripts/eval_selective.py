"""Selective-prediction evaluation across UQ methods.

Supports:
    --uq mc_dropout     (single checkpoint; T stochastic forward passes)
    --uq ensembles      (M checkpoints via --ensemble-checkpoints)
    --uq conformal      (single checkpoint; val set used for calibration)

Outputs per UQ method in `runs/<exp>/eval_<uq>/`:
    results.json        scalar metrics (acc, ECE, Brier, AURC, sel_acc@cov)
    risk_coverage.csv   per-coverage risk curve
    reliability.csv     binned confidence vs. accuracy
    predictions.csv     per-sample probs, confidence, label, pred

Usage:
    python scripts/eval_selective.py --config configs/mitbih_baseline.yaml \\
        --checkpoint runs/mitbih_baseline/best.pt --uq mc_dropout --mc-samples 30
    python scripts/eval_selective.py --config configs/mitbih_baseline.yaml \\
        --ensemble-checkpoints runs/ens/seed{1..5}/best.pt --uq ensembles
    python scripts/eval_selective.py --config configs/mitbih_baseline.yaml \\
        --checkpoint runs/mitbih_baseline/best.pt --uq conformal --alpha 0.1
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
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)
from rppg_sa.uncertainty.conformal import (
    calibrate_threshold,
    empirical_coverage,
    predict_sets,
    set_sizes,
)
from rppg_sa.uncertainty.ensembles import ensemble_predict
from rppg_sa.uncertainty.mc_dropout import mc_dropout_predict
from rppg_sa.utils.config import load_config
from rppg_sa.utils.seed import set_seed


# ----------------------------------------------------------------------
# device + data
# ----------------------------------------------------------------------
def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _build_loaders(cfg: dict[str, Any], run_dir: Path, batch_size: int = 64):
    """Return (val_loader, test_loader, label_names). Val is needed for conformal."""
    source = cfg["data"]["source"]
    if source != "mitbih":
        raise ValueError(f"Unsupported data source for eval: {source}")
    from rppg_sa.data.mitbih_torch import MITBIHSegmentDataset, subject_disjoint_split

    ds = MITBIHSegmentDataset(
        root=cfg["data"]["root"],
        target_fs=float(cfg["data"]["target_fs"]),
        window_seconds=float(cfg["data"]["window_seconds"]),
    )
    splits_file = run_dir / "splits.json"
    if splits_file.exists():
        with splits_file.open() as f:
            sp = json.load(f)
        val_idx = sp.get("val_idx", [])
        test_idx = sp["test_idx"]
    else:
        train_ids = [str(r) for r in cfg["data"]["splits"]["train"]]
        val_ids = [str(r) for r in cfg["data"]["splits"]["val"]]
        test_ids = [str(r) for r in cfg["data"]["splits"]["test"]]
        _, val_idx, test_idx = subject_disjoint_split(ds, train_ids, val_ids, test_ids)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return val_loader, test_loader, ds.LABEL_NAMES


# ----------------------------------------------------------------------
# model loading
# ----------------------------------------------------------------------
def _build_model(cfg: dict[str, Any], device: torch.device) -> CNN1DTransformer:
    mcfg = cfg["classifier"]
    model = CNN1DTransformer(
        in_channels=int(mcfg["in_channels"]),
        num_classes=int(mcfg["num_classes"]),
        conv_channels=tuple(mcfg["conv_channels"]),
        transformer_layers=int(mcfg["transformer_layers"]),
        transformer_heads=int(mcfg["transformer_heads"]),
        dropout=float(mcfg["dropout"]),
    ).to(device)
    return model


def _load_checkpoint(model: CNN1DTransformer, path: Path, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)


# ----------------------------------------------------------------------
# UQ-method-specific forward passes
# ----------------------------------------------------------------------
def _forward_mc_dropout(model, loader, device, T: int):
    probs_list, ent_list, mi_list, labels = [], [], [], []
    for batch in loader:
        x = batch["signal"].to(device)
        p, h, mi = mc_dropout_predict(model, x, num_samples=T)
        probs_list.append(p); ent_list.append(h); mi_list.append(mi)
        labels.extend(batch["label"].tolist())
    return (
        np.concatenate(probs_list, axis=0),
        np.concatenate(ent_list, axis=0),
        np.concatenate(mi_list, axis=0),
        np.array(labels, dtype=np.int64),
    )


def _forward_ensemble(models, loader, device):
    probs_list, ent_list, mi_list, labels = [], [], [], []
    for batch in loader:
        x = batch["signal"].to(device)
        p, h, mi = ensemble_predict(models, x)
        probs_list.append(p); ent_list.append(h); mi_list.append(mi)
        labels.extend(batch["label"].tolist())
    return (
        np.concatenate(probs_list, axis=0),
        np.concatenate(ent_list, axis=0),
        np.concatenate(mi_list, axis=0),
        np.array(labels, dtype=np.int64),
    )


@torch.no_grad()
def _forward_deterministic(model, loader, device):
    """Single-pass softmax with dropout OFF. Used as input to conformal."""
    model.eval()
    probs_list, labels = [], []
    for batch in loader:
        x = batch["signal"].to(device)
        logits = model(x)
        probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.extend(batch["label"].tolist())
    return np.concatenate(probs_list, axis=0), np.array(labels, dtype=np.int64)


# ----------------------------------------------------------------------
# reporting helpers
# ----------------------------------------------------------------------
def _reliability_table(probs, labels, n_bins: int = 15):
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        rows.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "n": int(in_bin.sum()),
            "mean_confidence": float(confs[in_bin].mean()) if in_bin.any() else float("nan"),
            "accuracy": float(correct[in_bin].mean()) if in_bin.any() else float("nan"),
        })
    return rows


def _per_class(probs, labels, label_names, num_classes, entropy=None):
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)
    out = {}
    for c in range(num_classes):
        m = labels == c
        if not m.any():
            continue
        entry = {"n": int(m.sum()), "accuracy": float(correct[m].mean())}
        if entropy is not None:
            entry["mean_entropy"] = float(entropy[m].mean())
        out[label_names[c]] = entry
    return out


def _write_outputs(out_dir, results, probs, labels, confidences, rc, label_names, extra_rows=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    with (out_dir / "risk_coverage.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["coverage", "risk"])
        for cov, r in zip(rc.coverage, rc.risk):
            w.writerow([f"{cov:.6f}", f"{r:.6f}"])

    with (out_dir / "reliability.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bin_lo", "bin_hi", "n", "mean_confidence", "accuracy"])
        w.writeheader()
        for row in _reliability_table(probs, labels):
            w.writerow(row)

    preds = probs.argmax(axis=1)
    fieldnames = ["idx", "label", "pred", "confidence"] + [f"p_{n}" for n in label_names]
    if extra_rows:
        fieldnames += list(extra_rows[0].keys())
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(fieldnames)
        for i, (lbl, prd, c, p) in enumerate(zip(labels, preds, confidences, probs)):
            row = [i, int(lbl), int(prd), f"{c:.6f}", *[f"{v:.6f}" for v in p]]
            if extra_rows:
                row += [extra_rows[i][k] for k in extra_rows[0].keys()]
            w.writerow(row)
    print(f"Wrote artefacts to {out_dir}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective prediction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Single checkpoint (mc_dropout, conformal).")
    parser.add_argument("--ensemble-checkpoints", type=Path, nargs="+", default=None,
                        help="M checkpoints for --uq ensembles.")
    parser.add_argument("--uq", choices=["mc_dropout", "ensembles", "conformal"], default="mc_dropout")
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Conformal miscoverage level; 0.1 → 90% sets.")
    parser.add_argument("--coverages", nargs="+", type=float,
                        default=[0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))
    device = _resolve_device(cfg["experiment"]["device"])
    print(f"Device: {device}")

    # ---------- Validate args ----------
    if args.uq in ("mc_dropout", "conformal") and args.checkpoint is None:
        raise ValueError(f"--uq {args.uq} requires --checkpoint")
    if args.uq == "ensembles" and not args.ensemble_checkpoints:
        raise ValueError("--uq ensembles requires --ensemble-checkpoints")

    # ---------- Data ----------
    primary_ckpt_dir = (
        args.checkpoint.parent if args.checkpoint else args.ensemble_checkpoints[0].parent
    )
    val_loader, test_loader, label_names = _build_loaders(cfg, primary_ckpt_dir)
    num_classes = int(cfg["classifier"]["num_classes"])

    # ---------- Dispatch on UQ method ----------
    if args.uq == "mc_dropout":
        model = _build_model(cfg, device)
        _load_checkpoint(model, args.checkpoint, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
        probs, entropy, mi, labels = _forward_mc_dropout(
            model, test_loader, device, T=args.mc_samples
        )
        confidences = -entropy
        extra = {"entropy": entropy, "mutual_info": mi}
        method_meta = {"mc_samples": args.mc_samples}

    elif args.uq == "ensembles":
        models = []
        for p in args.ensemble_checkpoints:
            m = _build_model(cfg, device)
            _load_checkpoint(m, p, device)
            m.eval()
            models.append(m)
        print(f"Loaded ensemble of {len(models)} models")
        probs, entropy, mi, labels = _forward_ensemble(models, test_loader, device)
        confidences = -entropy
        extra = {"entropy": entropy, "mutual_info": mi}
        method_meta = {"ensemble_size": len(models),
                       "checkpoints": [str(p) for p in args.ensemble_checkpoints]}

    else:  # conformal
        model = _build_model(cfg, device)
        _load_checkpoint(model, args.checkpoint, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
        cal_probs, cal_labels = _forward_deterministic(model, val_loader, device)
        probs, labels = _forward_deterministic(model, test_loader, device)
        q_hat = calibrate_threshold(cal_probs, cal_labels, alpha=args.alpha)
        sets = predict_sets(probs, q_hat)
        sizes = set_sizes(sets)
        cov = empirical_coverage(sets, labels)
        # Confidence for selective curve: 1 - non-conformity = top-class prob
        # (continuous; ties broken by full softmax).
        confidences = probs.max(axis=1)
        entropy = None
        mi = None
        extra = {"set_size": sizes.astype(np.float64)}
        method_meta = {
            "alpha": args.alpha,
            "q_hat": q_hat,
            "empirical_coverage": cov,
            "mean_set_size": float(sizes.mean()),
            "cal_n": int(len(cal_labels)),
        }
        print(f"Conformal: q_hat={q_hat:.4f}  empirical_coverage={cov:.4f}  "
              f"mean_set_size={sizes.mean():.3f}")

    # ---------- Metrics (common to all UQ methods) ----------
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)
    accuracy = float(correct.mean())
    ece = expected_calibration_error(probs, labels, n_bins=15)
    bs = brier_score(probs, labels, num_classes=num_classes)
    rc = risk_coverage_curve(confidences, correct)
    sel_acc = {f"sel_acc@{c}": selective_accuracy_at_coverage(confidences, correct, c)
               for c in args.coverages}

    results = {
        "uq_method": args.uq,
        **method_meta,
        "test_n": int(len(labels)),
        "accuracy": accuracy,
        "ece": ece,
        "brier_score": bs,
        "aurc": rc.aurc,
        **sel_acc,
        "per_class": _per_class(probs, labels, label_names, num_classes, entropy=entropy),
    }
    if entropy is not None:
        results["mean_predictive_entropy"] = float(entropy.mean())
    if mi is not None:
        results["mean_mutual_info"] = float(mi.mean())

    out_dir = primary_ckpt_dir / f"eval_{args.uq}"
    extra_rows = [
        {k: f"{extra[k][i]:.6f}" for k in extra}
        for i in range(len(labels))
    ] if extra else None
    _write_outputs(out_dir, results, probs, labels, confidences, rc, label_names, extra_rows)


if __name__ == "__main__":
    main()
