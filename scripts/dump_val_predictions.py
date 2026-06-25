"""Dump validation-set predictions in the same format as eval_selective.py's
predictions.csv. Needed for class-weight optimization (LW-CCSD).

Supports the same UQ methods as eval_selective.py:
    --uq deterministic  (default; single forward pass, dropout off)
    --uq mc_dropout     (single checkpoint; T stochastic passes; entropy-confidence)
    --uq ensembles      (M checkpoints via --ensemble-checkpoints; avg softmax)

Usage:
    python scripts/dump_val_predictions.py \\
        --config configs/synth_rppg_cinc.yaml \\
        --checkpoint runs/synth_rppg_cinc/best.pt
    python scripts/dump_val_predictions.py \\
        --config configs/synth_rppg_cinc.yaml \\
        --checkpoint runs/synth_rppg_cinc/best.pt \\
        --uq mc_dropout --mc-samples 30
    python scripts/dump_val_predictions.py \\
        --config configs/synth_rppg_cinc.yaml \\
        --uq ensembles --ensemble-checkpoints runs/synth_rppg_cinc_ens{1..5}/best.pt
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
from rppg_sa.models.edl_classifier import CNN1DTransformerEDL
from rppg_sa.models.sngp_classifier import CNN1DTransformerSNGP
from rppg_sa.uncertainty.ensembles import ensemble_predict
from rppg_sa.uncertainty.evidential import edl_uncertainty
from rppg_sa.uncertainty.mc_dropout import mc_dropout_predict
from rppg_sa.uncertainty.sngp import mean_field_logits
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
        val_idx = sp["val_idx"]
    else:
        train_ids = [str(r) for r in cfg["data"]["splits"]["train"]]
        val_ids = [str(r) for r in cfg["data"]["splits"]["val"]]
        test_ids = [str(r) for r in cfg["data"]["splits"]["test"]]
        _, val_idx, _ = subject_disjoint_split(ds, train_ids, val_ids, test_ids)

    loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, ds.LABEL_NAMES, [ds.signals[i] for i in val_idx]


def _build_model(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    mcfg = cfg["classifier"]
    head_type = mcfg.get("head", "softmax")
    if head_type == "evidential":
        return CNN1DTransformerEDL(
            in_channels=int(mcfg["in_channels"]),
            num_classes=int(mcfg["num_classes"]),
            conv_channels=tuple(mcfg["conv_channels"]),
            transformer_layers=int(mcfg["transformer_layers"]),
            transformer_heads=int(mcfg["transformer_heads"]),
            dropout=float(mcfg["dropout"]),
        ).to(device)
    if head_type == "sngp":
        sngp_kwargs = mcfg.get("sngp", {})
        return CNN1DTransformerSNGP(
            in_channels=int(mcfg["in_channels"]),
            num_classes=int(mcfg["num_classes"]),
            conv_channels=tuple(mcfg["conv_channels"]),
            transformer_layers=int(mcfg["transformer_layers"]),
            transformer_heads=int(mcfg["transformer_heads"]),
            dropout=float(mcfg["dropout"]),
            rff_features=int(sngp_kwargs.get("rff_features", 1024)),
            rff_kernel_scale=float(sngp_kwargs.get("rff_kernel_scale", 1.0)),
            gp_ridge=float(sngp_kwargs.get("gp_ridge", 1e-3)),
            spectral_norm_coefficient=float(sngp_kwargs.get("spectral_norm_coefficient", 0.95)),
        ).to(device)
    return CNN1DTransformer(
        in_channels=int(mcfg["in_channels"]),
        num_classes=int(mcfg["num_classes"]),
        conv_channels=tuple(mcfg["conv_channels"]),
        transformer_layers=int(mcfg["transformer_layers"]),
        transformer_heads=int(mcfg["transformer_heads"]),
        dropout=float(mcfg["dropout"]),
    ).to(device)


def _load_checkpoint(model, path: Path, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)


@torch.no_grad()
def _forward_deterministic(model, loader, device):
    model.eval()
    probs_list, labels = [], []
    for batch in loader:
        x = batch["signal"].to(device)
        logits = model(x)
        probs_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
        labels.extend(batch["label"].tolist())
    return np.concatenate(probs_list, axis=0), np.array(labels, dtype=np.int64), None


def _forward_mc_dropout(model, loader, device, T: int):
    probs_list, ent_list, labels = [], [], []
    for batch in loader:
        x = batch["signal"].to(device)
        p, h, _ = mc_dropout_predict(model, x, num_samples=T)
        probs_list.append(p)
        ent_list.append(h)
        labels.extend(batch["label"].tolist())
    return (
        np.concatenate(probs_list, axis=0),
        np.array(labels, dtype=np.int64),
        np.concatenate(ent_list, axis=0),
    )


def _forward_ensemble(models, loader, device):
    probs_list, ent_list, labels = [], [], []
    for batch in loader:
        x = batch["signal"].to(device)
        p, h, _ = ensemble_predict(models, x)
        probs_list.append(p)
        ent_list.append(h)
        labels.extend(batch["label"].tolist())
    return (
        np.concatenate(probs_list, axis=0),
        np.array(labels, dtype=np.int64),
        np.concatenate(ent_list, axis=0),
    )


@torch.no_grad()
def _forward_evidential(model, loader, device):
    """EDL forward pass on val. Returns probs, labels, dirichlet uncertainty."""
    model.eval()
    probs_list, unc_list, labels = [], [], []
    for batch in loader:
        x = batch["signal"].to(device)
        alpha = model(x)
        probs, u = edl_uncertainty(alpha)
        probs_list.append(probs.cpu().numpy())
        unc_list.append(u.cpu().numpy())
        labels.extend(batch["label"].tolist())
    return (
        np.concatenate(probs_list, axis=0),
        np.array(labels, dtype=np.int64),
        np.concatenate(unc_list, axis=0),
    )


@torch.no_grad()
def _forward_sngp(model, loader, device):
    """SNGP forward with mean-field correction. Returns probs, labels, entropy."""
    model.eval()
    probs_list, ent_list, labels = [], [], []
    for batch in loader:
        x = batch["signal"].to(device)
        mean_logits, variance = model(x)
        if variance is None:
            probs = torch.softmax(mean_logits, dim=-1)
        else:
            corrected = mean_field_logits(mean_logits, variance)
            probs = torch.softmax(corrected, dim=-1)
        eps = 1e-12
        ent = -(probs * torch.log(probs + eps)).sum(dim=-1)
        probs_list.append(probs.cpu().numpy())
        ent_list.append(ent.cpu().numpy())
        labels.extend(batch["label"].tolist())
    return (
        np.concatenate(probs_list, axis=0),
        np.array(labels, dtype=np.int64),
        np.concatenate(ent_list, axis=0),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Single checkpoint (deterministic, mc_dropout).")
    p.add_argument("--ensemble-checkpoints", type=Path, nargs="+", default=None,
                   help="M checkpoints for --uq ensembles.")
    p.add_argument("--uq", choices=["deterministic", "mc_dropout", "ensembles", "evidential", "evidential_ensemble", "sngp"],
                   default="deterministic")
    p.add_argument("--mc-samples", type=int, default=30)
    p.add_argument("--out-dir-name", type=str, default=None,
                   help="Override the output subdir name (default eval_val_<uq>).")
    args = p.parse_args()

    if args.uq in ("deterministic", "mc_dropout", "evidential", "sngp") and args.checkpoint is None:
        raise SystemExit(f"--uq {args.uq} requires --checkpoint")
    if args.uq in ("ensembles", "evidential_ensemble") and not args.ensemble_checkpoints:
        raise SystemExit(f"--uq {args.uq} requires --ensemble-checkpoints")

    cfg = load_config(args.config)
    device = _resolve_device(cfg["experiment"]["device"])
    print(f"Device: {device}  uq: {args.uq}")

    primary_dir = (
        args.checkpoint.parent if args.checkpoint else args.ensemble_checkpoints[0].parent
    )
    loader, label_names, _ = _build_val_loader(cfg, primary_dir)

    if args.uq == "deterministic":
        model = _build_model(cfg, device)
        _load_checkpoint(model, args.checkpoint, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
        probs, labels, entropy = _forward_deterministic(model, loader, device)
        confidence = probs.max(axis=1)
    elif args.uq == "mc_dropout":
        model = _build_model(cfg, device)
        _load_checkpoint(model, args.checkpoint, device)
        print(f"Loaded checkpoint: {args.checkpoint}  (T={args.mc_samples} samples)")
        probs, labels, entropy = _forward_mc_dropout(model, loader, device, T=args.mc_samples)
        # eval_selective writes -entropy as confidence for mc_dropout; mirror that here.
        confidence = -entropy
    elif args.uq == "evidential":
        cfg["classifier"]["head"] = "evidential"
        model = _build_model(cfg, device)
        _load_checkpoint(model, args.checkpoint, device)
        print(f"Loaded EDL checkpoint: {args.checkpoint}")
        probs, labels, uncertainty = _forward_evidential(model, loader, device)
        # Confidence: lower Dirichlet uncertainty = higher confidence.
        confidence = -uncertainty
        # For the predictions.csv 'entropy' column (consumed downstream by
        # LW-CCSD as the same-shape confidence rank), use predictive entropy.
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    elif args.uq == "sngp":
        cfg["classifier"]["head"] = "sngp"
        model = _build_model(cfg, device)
        _load_checkpoint(model, args.checkpoint, device)
        print(f"Loaded SNGP checkpoint: {args.checkpoint}")
        probs, labels, entropy = _forward_sngp(model, loader, device)
        confidence = -entropy
    elif args.uq == "evidential_ensemble":
        cfg["classifier"]["head"] = "evidential"
        models = []
        for ckpt in args.ensemble_checkpoints:
            m = _build_model(cfg, device)
            _load_checkpoint(m, ckpt, device)
            m.eval()
            models.append(m)
        print(f"Loaded EDL ensemble of {len(models)} models")
        probs_acc, unc_acc, labels = None, None, []
        with torch.no_grad():
            for batch in loader:
                x = batch["signal"].to(device)
                alphas = torch.stack([m(x) for m in models], dim=0)
                alpha_mean = alphas.mean(dim=0)
                probs_b, u_b = edl_uncertainty(alpha_mean)
                p_np, u_np = probs_b.cpu().numpy(), u_b.cpu().numpy()
                probs_acc = p_np if probs_acc is None else np.concatenate([probs_acc, p_np], axis=0)
                unc_acc = u_np if unc_acc is None else np.concatenate([unc_acc, u_np], axis=0)
                labels.extend(batch["label"].tolist())
        probs, uncertainty = probs_acc, unc_acc
        labels = np.array(labels, dtype=np.int64)
        confidence = -uncertainty
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    else:  # ensembles
        models = []
        for ckpt in args.ensemble_checkpoints:
            m = _build_model(cfg, device)
            _load_checkpoint(m, ckpt, device)
            m.eval()
            models.append(m)
        print(f"Loaded ensemble of {len(models)} models")
        probs, labels, entropy = _forward_ensemble(models, loader, device)
        confidence = -entropy

    preds = probs.argmax(axis=1)

    out_dir_name = args.out_dir_name or f"eval_val_{args.uq}"
    out_dir = primary_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["idx", "label", "pred", "confidence"] + [f"p_{n}" for n in label_names]
        if entropy is not None:
            header.append("entropy")
        writer.writerow(header)
        for i in range(len(labels)):
            row = [i, int(labels[i]), int(preds[i]), f"{confidence[i]:.6f}",
                   *[f"{v:.6f}" for v in probs[i]]]
            if entropy is not None:
                row.append(f"{entropy[i]:.6f}")
            writer.writerow(row)
    print(f"Wrote {csv_path}  (n={len(labels)})")


if __name__ == "__main__":
    main()
