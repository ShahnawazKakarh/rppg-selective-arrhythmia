"""Train the rhythm classifier from a YAML config.

Usage:
    python scripts/train_classifier.py --config configs/baseline_physnet.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rPPG arrhythmia classifier.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    args = parser.parse_args()

    raise NotImplementedError(
        f"Trainer stub for config {args.config}. "
        "Wire up data loaders, model, optimizer, W&B logging."
    )


if __name__ == "__main__":
    main()
