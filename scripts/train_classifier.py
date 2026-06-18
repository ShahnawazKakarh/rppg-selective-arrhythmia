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
                seed=int(cfg["experiment"]["seed"]),
            )
            print(f"auto-split subjects: train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")
        train_idx, val_idx, test_idx = subject_disjoint_split(ds, train_ids, val_ids, test_ids)
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
    criterion: nn.Module,
    device: torch.device,
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
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        all_preds.extend(logits.argmax(-1).detach().cpu().tolist())
        all_labels.extend(y.detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(np.mean(np.array(all_preds) == np.array(all_labels))),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train arrhythmia classifier.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["experiment"]["seed"]))
    device = _resolve_device(cfg["experiment"]["device"])
    print(f"Device: {device}")

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
    model = CNN1DTransformer(
        in_channels=int(mcfg["in_channels"]),
        num_classes=int(mcfg["num_classes"]),
        conv_channels=tuple(mcfg["conv_channels"]),
        transformer_layers=int(mcfg["transformer_layers"]),
        transformer_heads=int(mcfg["transformer_heads"]),
        dropout=float(mcfg["dropout"]),
    ).to(device)

    # ---------- Loss / optimizer ----------
    train_labels = [full_ds.labels[i] for i in train_idx]
    if cfg["training"].get("class_weighted_loss", False):
        weights = _class_weights(train_labels, int(mcfg["num_classes"])).to(device)
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
        train_metrics = _epoch_loop(model, train_loader, optimizer, criterion, device)
        val_metrics = _epoch_loop(model, val_loader, None, criterion, device)
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
