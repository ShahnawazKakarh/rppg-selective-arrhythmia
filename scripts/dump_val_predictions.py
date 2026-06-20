"""Dump validation-set predictions in the same format as eval_selective.py's
predictions.csv. Needed for class-weight optimization (LW-CCSD).

Usage:
    python scripts/dump_val_predictions.py \
        --config configs/synth_rppg_cinc.yaml \
        --checkpoint runs/synth_rppg_cinc/best.pt
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
from rppg_sa.utils.config import load_config


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _build_val_loader(cfg: dict[str, Any], run_dir: Path, batch_size: int = 64):
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
        val_idx = sp["val_idx"]
    else:
        train_ids = [str(r) for r in cfg["data"]["splits"]["train"]]
        val_ids = [str(r) for r in cfg["data"]["splits"]["val"]]
        test_ids = [str(r) for r in cfg["data"]["splits"]["test"]]
        _, val_idx, _ = subject_disjoint_split(ds, train_ids, val_ids, test_ids)

    loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, ds.LABEL_NAMES, [ds.signals[i] for i in val_idx]


@torch.no_grad()
def _forward(model, loader, device):
    model.eval()
    probs_list, labels = [], []
    for batch in loader:
        x = batch["signal"].to(device)
        logits = model(x)
        probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.extend(batch["label"].tolist())
    return np.concatenate(probs_list, axis=0), np.array(labels, dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(cfg["experiment"]["device"])
    print(f"Device: {device}")

    loader, label_names, val_signals = _build_val_loader(cfg, args.checkpoint.parent)

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

    probs, labels = _forward(model, loader, device)
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    out_dir = args.checkpoint.parent / "eval_val_deterministic"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label", "pred", "confidence"] + [f"p_{n}" for n in label_names])
        for i, (lbl, prd, c, pr) in enumerate(zip(labels, preds, confidence, probs)):
            writer.writerow([i, int(lbl), int(prd), f"{c:.6f}", *[f"{v:.6f}" for v in pr]])
    print(f"Wrote {csv_path}  (n={len(labels)})")


if __name__ == "__main__":
    main()
