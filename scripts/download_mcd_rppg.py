"""Download MCD-rPPG dataset from Hugging Face Hub.

Dataset: milai-oks-sakura/mcd_rppg
  3,600 synchronized video recordings from 600 subjects with PPG, ECG, and
  extended health biomarkers (BP, SpO2, glucose, etc.). Used as the primary
  source for rPPG extractor training and pipeline validation.

  The original upload location `kyegorov/mcd_rppg` no longer resolves (404);
  `milai-oks-sakura/mcd_rppg` is the working mirror (CC-BY-4.0, 135 GB,
  gated: requires accepting a contact-sharing agreement on the dataset page
  while signed in to Hugging Face).

Companion repo: https://github.com/ksyegorov/mcd_rppg

Usage:
    # Full dataset (large — ~135 GB, runs in background)
    python scripts/download_mcd_rppg.py --out data/raw/mcd_rppg

    # Only metadata + a couple of subjects (smoke test)
    python scripts/download_mcd_rppg.py --out data/raw/mcd_rppg --allow-patterns "*.json" "subject_001/*" "subject_002/*"
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MCD-rPPG dataset from Hugging Face.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/mcd_rppg"),
        help="Output directory.",
    )
    parser.add_argument(
        "--repo-id",
        default="milai-oks-sakura/mcd_rppg",
        help="Hugging Face dataset repo ID.",
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="*",
        default=None,
        help="Optional glob patterns to restrict what is downloaded "
        "(useful for a partial smoke-test download).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (or set HF_TOKEN env var). "
        "Only required if the dataset is gated.",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id} -> {args.out}")
    if args.allow_patterns:
        print(f"  patterns: {args.allow_patterns}")

    local = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(args.out),
        allow_patterns=args.allow_patterns,
        token=args.token,
    )
    print(f"Downloaded to: {local}")

    # Quick summary.
    out_path = Path(local)
    n_files = sum(1 for _ in out_path.rglob("*") if _.is_file())
    total_mb = sum(p.stat().st_size for p in out_path.rglob("*") if p.is_file()) / (1024 * 1024)
    print(f"Summary: {n_files} files, {total_mb:.1f} MB total under {out_path}")


if __name__ == "__main__":
    main()
