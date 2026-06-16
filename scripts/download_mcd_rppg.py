"""Download MCD-rPPG dataset from Hugging Face.

Dataset: kyegorov/mcd_rppg
  3,600 synchronized video recordings from 600 subjects with PPG, ECG, and
  extended health biomarkers (BP, SpO2, stress, etc.). Used as the primary
  source for rPPG extractor training and pipeline validation.

Usage:
    python scripts/download_mcd_rppg.py --out data/raw/mcd_rppg
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MCD-rPPG dataset.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/mcd_rppg"),
        help="Output directory.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Implementation note:
    # Use `huggingface_hub.snapshot_download(repo_id="kyegorov/mcd_rppg",
    #     repo_type="dataset", local_dir=args.out)` once the loader is wired up.
    raise NotImplementedError(
        "MCD-rPPG downloader stub. Wire up huggingface_hub.snapshot_download."
    )


if __name__ == "__main__":
    main()
