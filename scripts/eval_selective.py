"""Evaluate selective-prediction performance.

Computes risk-coverage curves, AURC, selective accuracy at target coverages,
and calibration error for a trained checkpoint.

Usage:
    python scripts/eval_selective.py --config configs/selective_mcdropout.yaml \\
        --checkpoint runs/mcdropout/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective prediction.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint.")
    args = parser.parse_args()

    raise NotImplementedError(
        f"Selective-eval stub for {args.checkpoint} with config {args.config}."
    )


if __name__ == "__main__":
    main()
