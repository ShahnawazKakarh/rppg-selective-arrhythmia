"""Download rPPG-10 dataset from Mendeley Data.

Dataset: rPPG-10 (Mendeley DOI 10.17632/bx8982xgwt.1)
  26 Portuguese university students, 10-minute facial recordings split into
  three ROIs (forehead, left cheek, right cheek) with synchronized ECG.
  Used as an independent validation set for the rPPG extractor.

Usage:
    python scripts/download_rppg10.py --out data/raw/rppg10
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download rPPG-10 dataset.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/rppg10"),
        help="Output directory.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "rPPG-10 downloader stub. Mendeley does not expose a stable API; "
        "fetch the archive URL from the dataset page and stream it here."
    )


if __name__ == "__main__":
    main()
