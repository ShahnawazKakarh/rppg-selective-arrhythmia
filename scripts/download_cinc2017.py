"""Download the PhysioNet/CinC 2017 AF Classification Challenge training set.

8,528 single-lead ECG recordings (~300 Hz, 9-60 s each, ~1.5 GB zipped).
REFERENCE.csv labels each record as N (normal), A (atrial fibrillation),
O (other rhythm), or ~ (noisy/too-noisy-to-classify).

Source: https://physionet.org/content/challenge-2017/1.0.0/

Usage:
    python scripts/download_cinc2017.py --out data/raw/cinc2017
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


TRAINING_ZIP_URL = (
    "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/cinc2017"))
    parser.add_argument("--keep-zip", action="store_true",
                        help="Don't delete the downloaded zip after extraction.")
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "training2017.zip"
    extracted_marker = out / "training2017" / "REFERENCE.csv"

    if extracted_marker.exists():
        print(f"Already extracted: {extracted_marker.parent}")
        return

    if not zip_path.exists():
        print(f"Downloading {TRAINING_ZIP_URL}")
        print("  ~1.5 GB; this may take a few minutes ...")
        with urllib.request.urlopen(TRAINING_ZIP_URL) as resp, zip_path.open("wb") as f:
            shutil.copyfileobj(resp, f)
        print(f"  saved to {zip_path}")
    else:
        print(f"Using existing zip: {zip_path}")

    print(f"Extracting to {out} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)

    if not extracted_marker.exists():
        raise RuntimeError(
            f"Extraction finished but {extracted_marker} not found; "
            "check the archive layout."
        )

    # Quick label distribution summary.
    counts: dict[str, int] = {}
    with extracted_marker.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, label = line.split(",", 1)
            counts[label] = counts.get(label, 0) + 1
    print(f"Extracted {sum(counts.values())} records:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")

    if not args.keep_zip:
        zip_path.unlink()
        print(f"Removed {zip_path}")


if __name__ == "__main__":
    main()
