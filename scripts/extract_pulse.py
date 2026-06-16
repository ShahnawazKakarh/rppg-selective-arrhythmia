"""Extract pulse waveform from face video.

Supports classical (CHROM, POS) and learned (PhysNet) extractors.

Usage:
    python scripts/extract_pulse.py --video VIDEO --method chrom --out pulse.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract rPPG pulse from face video.")
    parser.add_argument("--video", type=Path, required=True, help="Input video file.")
    parser.add_argument(
        "--method",
        choices=["chrom", "pos", "physnet"],
        default="chrom",
        help="rPPG extraction method.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .npy waveform.")
    args = parser.parse_args()

    raise NotImplementedError(
        f"Extractor stub. Implement {args.method.upper()} in src/rppg_sa/extractors/."
    )


if __name__ == "__main__":
    main()
