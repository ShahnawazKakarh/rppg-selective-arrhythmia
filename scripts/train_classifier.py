"""Real training loop.

Wires together: config loading, MIT-BIH (or rPPG-segment) dataset, the
CNN1D-Transformer classifier, AdamW + cosine schedule, class-weighted
cross-entropy, validation tracking with macro-F1, early stopping, optional W&B,
and checkpointing.

Usage:
    python scripts/train_classifier.py --config configs/mitbih_baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset

from rppg_sa.models.cnn1d_transformer import CNN1DTransformer
from rppg_sa.models.edl_classifier import CNN1DTransformerEDL
from rppg_sa.uncertainty.evidential import edl_mse_loss
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


def _build_dataset(cfg: dict[str, Any]):
    source = cfg["data"]["source"]
    if source == "mitbih":
        from rppg_sa.data.mitbih_torch import MITBIHSegmentDataset, subject_disjoint_split

        ds = MITBIHSegmentDataset(
            root=cfg["data"]["root"],
            target_fs=float(cfg["data"]["target_fs"]),
            window_seconds=float(cfg["data"]["window_seconds"]),
        )
        train_ids = [str(r) for r in cfg["data"]["splits"]["train"]]
        val_ids = [str(r) for r in cfg["data"]["splits"]["val"]]
        test_ids = [str(r) for r in cfg["data"]["splits"]["test"]]
        train_idx, val_idx, test_idx = subject_disjoint_split(ds, train_ids, val_ids, test_ids)
        return ds, train_idx, val_idx, test_idx, ds.LABEL_NAMES
    if source == "synth_rppg":
        from rppg_sa.data.synth_rppg_torch import (
            SynthRPPGSegmentDataset,
            subject_disjoint_split as synth_split,
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
        train_ids = [str(r) for r in cfg["data"]["splits"]["train"]]
        val_ids = [str(r) for r in cfg["data"]["splits"]["val"]]
        test_ids = [str(r) for r in cfg["data"]["splits"]["test"]]
        train_idx, val_idx, test_idx = synth_split(ds, train_ids, val_ids, test_ids)
        return ds, train_idx, val_idx, test_idx, ds.LABEL_NAMES
    if source == "mcd_rppg":
        from rppg_sa.data.mcd_rppg_torch import (
            MCDRPPGSegmentDataset,
            auto_split_subjects,
            subject_disjoint_split,
        )

        ds = MCDRPPGSegmentDataset(
            root=cfg["data"]["root"],
            cache_dir=cfg["data"].get("cache_dir"),
            camera=cfg["data"].get("camera", "FullHDwebcam"),
            view=cfg["data"].get("view", "front"),
            target_fs=float(cfg["data"]["target_fs"]),
            window_seconds=float(cfg["data"]["window_seconds"]),
            max_records=cfg["data"].get("max_records"),
        )
        splits = cfg["data"].get("splits")
        if splits and "train" in splits:
            train_ids = [str(r) for r in splits["train"]]
            val_ids = [str(r) for r in splits["val"]]
            test_ids = [str(r) for r in splits["test"]]
        else:
            # Deterministic 70/15/15 subject-level auto-split.
            train_ids, val_ids, test_ids = auto_split_subjects(
                ds,
                val_frac=float(cfg["data"].get("val_frac", 0.15)),
                test_frac=float(cfg["data"].get("test_frac", 0.15)),
                seed=int(cfg["experiment"]["data_seed"]),
            )
            print(f"auto-split subjects (data_seed={cfg['experiment']['data_seed']}): "
                  f"train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")
        train_idx, val_idx, test_idx = subject_disjoint_split(ds, train_ids, val_ids, test_ids)
        return ds, train_idx, val_idx, test_idx, ds.LABEL_NAMES
    if source == "synth_rppg_cinc":
        from rppg_sa.data.cinc2017_synth_torch import (
            CinCSynthRPPGSegmentDataset,
            auto_split_records,
            subject_disjoint_split as cinc_split,
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
        splits = cfg["data"].get("splits")
        if splits and "train" in splits:
            train_ids = [str(r) for r in splits["train"]]
            val_ids = [str(r) for r in splits["val"]]
            test_ids = [str(r) for r in splits["test"]]
        else:
            train_ids, val_ids, test_ids = auto_split_records(
                cfg["data"]["root"],
                val_frac=float(cfg["data"].get("val_frac", 0.15)),
                test_frac=float(cfg["data"].get("test_frac", 0.15)),
                seed=int(cfg["experiment"]["data_seed"]),
            )
            print(f"auto-split records (data_seed={cfg['experiment']['data_seed']}): "
                  f"train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")
        train_idx, val_idx, test_idx = cinc_split(ds, train_ids, val_ids, test_ids)
        return ds, train_idx, val_idx, test_idx, ds.LABEL_NAMES
    raise ValueError(f"Unsupported data source: {source}")


def _class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        n_c = counts.get(c, 0)
        # Inverse-frequency weighting with a small floor to avoid blowups.
        w = total / (num_classes * max(n_c, 1))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def _epoch_loop(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module | None,
    device: torch.device,
    head_type: str = "softmax",
    edl_epoch: int = 0,
    edl_annealing_steps: int = 10,
    edl_num_classes: int = 3,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    losses: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        x = batch["signal"].to(device)
        y = batch["label"].to(device)
        with torch.set_grad_enabled(train_mode):
            out = model(x)
            if head_type == "evidential":
                # `out` is alpha; predictions are argmax of expected probabilities
                # p = alpha / S, equivalent to argmax of alpha.
                loss = edl_mse_loss(
                    out, y, edl_num_classes,
                    epoch=edl_epoch, annealing_steps=edl_annealing_steps,
                )
                preds = out.argmax(-1)
            else:
                assert criterion is not None, "softmax head requires criterion"
                loss = criterion(out, y)
                preds = out.argmax(-1)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(np.array(all_preds) == np.array(all_labels))),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train arrhythmia classifier.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Legacy: set both data_seed and model_seed to this value. "
                             "Use --data-seed / --model-seed for the decoupled methodology.")
    parser.add_argument("--data-seed", type=int, default=None,
                        help="Seed for train/val/test split. Same across ensemble members.")
    parser.add_argument("--model-seed", type=int, default=None,
                        help="Seed for model initialisation + DataLoader shuffle. "
                             "Differs across ensemble members.")
    parser.add_argument("--name-suffix", type=str, default="",
                        help="Append to experiment.name; lands in runs/<name><suffix>/.")
    parser.add_argument("--head", type=str, default=None, choices=["softmax", "evidential"],
                        help="Override classifier.head (default reads from config; "
                             "falls back to 'softmax').")
    parser.add_argument("--edl-annealing-steps", type=int, default=10,
                        help="Epochs to anneal the EDL KL weight from 0 to 1.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Resolve data_seed and model_seed with backwards compatibility.
    legacy_seed = int(cfg["experiment"].get("seed", 0))
    data_seed = args.data_seed
    model_seed = args.model_seed
    if args.seed is not None:
        # Legacy single-seed path: both seeds default to this value, but explicit
        # --data-seed / --model-seed still take precedence.
        if data_seed is None:
            data_seed = args.seed
        if model_seed is None:
            model_seed = args.seed
    if data_seed is None:
        data_seed = int(cfg["experiment"].get("data_seed", legacy_seed))
    if model_seed is None:
        model_seed = int(cfg["experiment"].get("model_seed", legacy_seed))

    cfg["experiment"]["data_seed"] = int(data_seed)
    cfg["experiment"]["model_seed"] = int(model_seed)
    # Keep legacy field present so downstream tooling that reads experiment.seed
    # still finds a value; resolve it to model_seed since that affects torch state.
    cfg["experiment"]["seed"] = int(model_seed)
    if args.name_suffix:
        cfg["experiment"]["name"] = cfg["experiment"]["name"] + args.name_suffix

    set_seed(int(model_seed))
    device = _resolve_device(cfg["experiment"]["device"])
    print(f"Device: {device}  data_seed={data_seed}  model_seed={model_seed}")

    # ---------- Data ----------
    full_ds, train_idx, val_idx, test_idx, label_names = _build_dataset(cfg)
    print(f"Dataset: {len(full_ds)} segments | classes: {label_names}")
    print(f"Class counts: {full_ds.class_counts()}")
    print(f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---------- Model ----------
    mcfg = cfg["classifier"]
    head_type = args.head if args.head is not None else mcfg.get("head", "softmax")
    num_classes = int(mcfg["num_classes"])
    if head_type == "evidential":
        model = CNN1DTransformerEDL(
            in_channels=int(mcfg["in_channels"]),
            num_classes=num_classes,
            conv_channels=tuple(mcfg["conv_channels"]),
            transformer_layers=int(mcfg["transformer_layers"]),
            transformer_heads=int(mcfg["transformer_heads"]),
            dropout=float(mcfg["dropout"]),
        ).to(device)
        print(f"Head: evidential (Dirichlet, EDL); KL annealing over {args.edl_annealing_steps} epochs")
    else:
        model = CNN1DTransformer(
            in_channels=int(mcfg["in_channels"]),
            num_classes=num_classes,
            conv_channels=tuple(mcfg["conv_channels"]),
            transformer_layers=int(mcfg["transformer_layers"]),
            transformer_heads=int(mcfg["transformer_heads"]),
            dropout=float(mcfg["dropout"]),
        ).to(device)
        print("Head: softmax (cross-entropy)")
    cfg["classifier"]["head"] = head_type

    # ---------- Loss / optimizer ----------
    train_labels = [full_ds.labels[i] for i in train_idx]
    if head_type == "evidential":
        # EDL uses its own loss (edl_mse_loss); CE criterion not needed.
        criterion = None
        print("Loss: EDL Type-II MSE + annealed KL (class weights ignored under EDL)")
    elif cfg["training"].get("class_weighted_loss", False):
        weights = _class_weights(train_labels, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Class weights: {weights.cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    epochs = int(cfg["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---------- W&B (optional) ----------
    run = None
    if cfg["logging"].get("use_wandb", False):
        import wandb

        run = wandb.init(
            project=cfg["logging"]["project"],
            entity=cfg["logging"].get("entity"),
            name=cfg["experiment"]["name"],
            config=cfg,
        )

    # ---------- Checkpoint dir ----------
    log_root = Path(cfg["logging"].get("log_dir", "runs")) / cfg["experiment"]["name"]
    log_root.mkdir(parents=True, exist_ok=True)
    best_metric = -math.inf
    patience = int(cfg["training"].get("early_stopping_patience", 0))
    stale = 0
    history: list[dict[str, Any]] = []

    # ---------- Training ----------
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch_loop(
            model, train_loader, optimizer, criterion, device,
            head_type=head_type, edl_epoch=epoch,
            edl_annealing_steps=args.edl_annealing_steps,
            edl_num_classes=num_classes,
        )
        val_metrics = _epoch_loop(
            model, val_loader, None, criterion, device,
            head_type=head_type, edl_epoch=epoch,
            edl_annealing_steps=args.edl_annealing_steps,
            edl_num_classes=num_classes,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:3d} | "
            f"train loss {train_metrics['loss']:.4f} f1 {train_metrics['macro_f1']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} f1 {val_metrics['macro_f1']:.3f}"
        )
        if run is not None:
            run.log(row, step=epoch)

        metric = val_metrics["macro_f1"]
        if metric > best_metric:
            best_metric = metric
            stale = 0
            ckpt_path = log_root / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": metric,
                    "cfg": cfg,
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path} (val_macro_f1={metric:.4f})")
        else:
            stale += 1
            if patience and stale >= patience:
                print(f"early stopping after {patience} stale epochs")
                break

    # ---------- Persist history ----------
    with (log_root / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    # Also persist the indices used in this run so eval is reproducible.
    with (log_root / "splits.json").open("w") as f:
        json.dump(
            {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx},
            f,
        )
    print(f"Training complete. Best val_macro_f1={best_metric:.4f}. Artefacts in {log_root}.")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
