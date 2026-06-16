"""Download the MIT-BIH Arrhythmia Database from PhysioNet.

Uses `wfdb.dl_database` which streams from physionet.org. The full database is
~75 MB. Records 102, 104, 107, 217 have paced beats; 100..234 with gaps.

Reference:
    Moody & Mark (2001). "The impact of the MIT-BIH Arrhythmia Database."
    https://physionet.org/content/mitdb/1.0.0/
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MIT-BIH Arrhythmia Database.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/mitbih"),
        help="Output directory.",
    )
    parser.add_argument(
        "--records",
        nargs="*",
        default=None,
        help="Optional subset of record IDs (e.g. 100 101 ... 234). "
        "Default: all 48 standard records.",
    )
    args = parser.parse_args()

    try:
        import wfdb
    except ImportError as e:
        raise SystemExit("wfdb is required. Install with: pip install wfdb") from e

    args.out.mkdir(parents=True, exist_ok=True)

    records = args.records
    if records is None:
        # wfdb.dl_database requires an explicit list; fetch the canonical one.
        records = wfdb.get_record_list("mitdb")
        print(f"Fetched record list: {len(records)} records.")

    print(f"Downloading MIT-BIH to {args.out} ...")
    wfdb.dl_database("mitdb", str(args.out), records=records, keep_subdirs=False)

    n_hea = len(list(args.out.glob("*.hea")))
    print(f"Done. {n_hea} record headers in {args.out}.")


if __name__ == "__main__":
    main()
