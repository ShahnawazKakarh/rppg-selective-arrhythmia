"""Signal-quality-aware deferral evaluation.

Headline methodological claim of the paper: combining model uncertainty with
physical signal-quality (spectral SNR + template SQI) beats UQ-only deferral.

For a fixed checkpoint, this script:
  1. Runs deterministic forward pass on the test set to get softmax probs.
  2. Computes per-segment SNR (dB) and template SQI from the raw pulse.
  3. Sweeps quality_weight w ∈ {0, 0.15, 0.3, 0.5, 0.7, 1.0}, where
        combined_score(w) = (1-w) * model_confidence + w * signal_quality
     (each rank-normalized). w=0 is the UQ-only baseline.
  4. Computes risk-coverage curve, AURC, and selective accuracy at
     {0.5, 0.7, 0.8, 0.9, 0.95, 1.0} for each w.
  5. Writes a single CSV summarising sweep and JSON with the per-w details.

Usage:
    python scripts/eval_sqi_deferral.py \
        --config configs/synth_rppg_cinc.yaml \
        --checkpoint runs/synth_rppg_cinc/best.pt \
        --signal-quality template_sqi
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

from rppg_sa.extractors.signal_quality import summarize_quality
from rppg_sa.models.cnn1d_transformer import CNN1DTransformer
from rppg_sa.selective.deferral import combine_score, select_at_coverage
from rppg_sa.selective.metrics import (
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)
from rppg_sa.utils.config import load_config
from rppg_sa.utils.seed import set_seed


# Defaults shared with eval_selective.py — keep them in sync.
DEFAULT_WEIGHTS = [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]
DEFAULT_COVERAGES = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0]


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _build_test_loader(cfg: dict[str, Any], run_dir: Path, batch_size: int = 64):
    """Returns (test_loader, label_names, test_signals_at_fs, fs)."""
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

    loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    test_signals = [ds.signals[i] for i in test_idx]
    fs = float(cfg["data"]["target_fs"])
    return loader, ds.LABEL_NAMES, test_signals, fs


def _build_model(cfg: dict[str, Any], device: torch.device) -> CNN1DTransformer:
    mcfg = cfg["classifier"]
    return CNN1DTransformer(
        in_channels=int(mcfg["in_channels"]),
        num_classes=int(mcfg["num_classes"]),
        conv_channels=tuple(mcfg["conv_channels"]),
        transformer_layers=int(mcfg["transformer_layers"]),
        transformer_heads=int(mcfg["transformer_heads"]),
        dropout=float(mcfg["dropout"]),
    ).to(device)


@torch.no_grad()
def _forward_deterministic(model, loader, device):
    model.eval()
    probs_list, labels = [], []
    for batch in loader:
        x = batch["signal"].to(device)
        logits = model(x)
        probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.extend(batch["label"].tolist())
    return np.concatenate(probs_list, axis=0), np.array(labels, dtype=np.int64)


def _compute_quality(signals: list[np.ndarray], fs: float, key: str) -> np.ndarray:
    out = np.empty(len(signals), dtype=np.float32)
    for i, seg in enumerate(signals):
        q = summarize_quality(seg, fs=fs)
        out[i] = q[key]
    return out


def _load_predictions_csv(path: Path):
    """Read predictions.csv produced by eval_selective.py.

    Returns (labels, preds, confidence, run_dir).
    """
    import csv as _csv
    labels: list[int] = []
    preds: list[int] = []
    confs: list[float] = []
    with path.open() as f:
        reader = _csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            preds.append(int(row["pred"]))
            confs.append(float(row["confidence"]))
    return (
        np.array(labels, dtype=np.int64),
        np.array(preds, dtype=np.int64),
        np.array(confs, dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Single-model checkpoint; runs deterministic forward pass.")
    parser.add_argument("--predictions-csv", type=Path, default=None,
                        help="Path to existing eval_*/predictions.csv; reuses its "
                             "confidences (entropy-based for MC Dropout, averaged "
                             "softmax max for ensembles, etc.).")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run dir containing splits.json. Required with "
                             "--predictions-csv; ignored otherwise.")
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix appended to output dir name (e.g. _mc_dropout).")
    parser.add_argument("--signal-quality", default="template_sqi",
                        choices=["snr_db", "template_sqi"])
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--coverages", type=float, nargs="+", default=DEFAULT_COVERAGES)
    args = parser.parse_args()

    if args.checkpoint is None and args.predictions_csv is None:
        raise SystemExit("Provide --checkpoint OR --predictions-csv")
    if args.predictions_csv and args.run_dir is None:
        raise SystemExit("--predictions-csv requires --run-dir (for splits.json)")

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))
    device = _resolve_device(cfg["experiment"]["device"])
    print(f"Device: {device}")

    if args.checkpoint is not None:
        loader, label_names, test_signals, fs = _build_test_loader(cfg, args.checkpoint.parent)
        model = _build_model(cfg, device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"Loaded checkpoint: {args.checkpoint}")

        probs, labels = _forward_deterministic(model, loader, device)
        preds = probs.argmax(axis=1)
        model_confidence = probs.max(axis=1)
        out_root = args.checkpoint.parent
    else:
        # Read confidences from an existing eval_*/predictions.csv (UQ output).
        _, label_names, test_signals, fs = _build_test_loader(cfg, args.run_dir)
        labels, preds, model_confidence = _load_predictions_csv(args.predictions_csv)
        print(
            f"Loaded {len(labels)} predictions from {args.predictions_csv} "
            f"(confidence range {model_confidence.min():.3f} … {model_confidence.max():.3f})"
        )
        if len(labels) != len(test_signals):
            raise SystemExit(
                f"length mismatch: predictions={len(labels)} vs test signals={len(test_signals)}. "
                f"Use the --run-dir that produced the predictions.csv."
            )
        out_root = args.predictions_csv.parent

    correct = (preds == labels).astype(np.float64)
    print(f"Computing {args.signal_quality} for {len(test_signals)} segments ...")
    sq = _compute_quality(test_signals, fs=fs, key=args.signal_quality)

    rows: list[dict[str, Any]] = []
    per_w_detail: dict[str, Any] = {}
    for w in args.weights:
        combined = combine_score(model_confidence, sq, quality_weight=w)
        rc = risk_coverage_curve(combined, correct)
        sel_acc = {
            f"sel_acc@{c}": selective_accuracy_at_coverage(combined, correct, c)
            for c in args.coverages
        }
        rows.append({"quality_weight": w, "aurc": rc.aurc, **sel_acc})
        per_w_detail[f"w={w}"] = {
            "aurc": rc.aurc,
            **sel_acc,
            "coverage_grid": [float(c) for c in rc.coverage],
            "risk_grid": [float(r) for r in rc.risk],
        }

    # Summary text + CSV + JSON outputs in a sibling dir to the checkpoint.
    out_dir = out_root / f"eval_sqi_deferral{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"sweep_{args.signal_quality}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "predictions_csv": str(args.predictions_csv) if args.predictions_csv else None,
        "signal_quality": args.signal_quality,
        "weights": [float(w) for w in args.weights],
        "coverages": [float(c) for c in args.coverages],
        "n_test": int(len(labels)),
        "accuracy": float(correct.mean()),
        "model_confidence_stats": {
            "mean": float(model_confidence.mean()),
            "std": float(model_confidence.std()),
        },
        "signal_quality_stats": {
            "mean": float(sq.mean()),
            "std": float(sq.std()),
            "min": float(sq.min()),
            "max": float(sq.max()),
        },
        "per_weight": per_w_detail,
    }
    with (out_dir / f"results_{args.signal_quality}.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Console table.
    print(f"\nSweep over {args.signal_quality}:")
    print(f"{'w':>5}  {'AURC':>7}  " + "  ".join(f"sel@{c:>4}" for c in args.coverages))
    for row in rows:
        sel_vals = "  ".join(f"{row[f'sel_acc@{c}']:>7.4f}" for c in args.coverages)
        print(f"{row['quality_weight']:>5.2f}  {row['aurc']:>7.4f}  {sel_vals}")

    best = min(rows, key=lambda r: r["aurc"])
    baseline = next(r for r in rows if r["quality_weight"] == 0.0)
    delta = baseline["aurc"] - best["aurc"]
    print(
        f"\nBest AURC at w={best['quality_weight']:.2f}: {best['aurc']:.4f}  "
        f"(vs UQ-only baseline {baseline['aurc']:.4f}, Δ={delta:+.4f})"
    )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
