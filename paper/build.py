"""Build the LW-CCSD paper PDF using WeasyPrint.

Mirrors the report/build.py pattern from retinal-selective-prediction.

Usage:
    pip install 'weasyprint>=63'
    python paper/build.py

Output: paper/lw-ccsd-rppg-af-v1.0.0.pdf
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
HTML_IN = PAPER_DIR / "report.html"
CSS_IN = PAPER_DIR / "style.css"
PDF_OUT = PAPER_DIR / "lw-ccsd-rppg-af-v1.6.0.pdf"


def main() -> int:
    try:
        from weasyprint import CSS, HTML
    except ImportError:
        sys.exit("Missing dependency. Install with: pip install 'weasyprint>=63'")

    for p in (HTML_IN, CSS_IN):
        if not p.exists():
            sys.exit(f"Missing {p}")

    # Verify figure PNGs exist; the report references them via relative paths.
    figs_dir = PAPER_DIR / "figures"
    for fig in ("naive_sqi_failure.png", "pareto_frontier.png", "snr_distribution.png"):
        if not (figs_dir / fig).exists():
            sys.exit(
                f"Missing figure: {figs_dir / fig}\n"
                f"  Generate first: python scripts/generate_paper_figures.py"
            )

    print(f"Building PDF from {HTML_IN}", flush=True)
    HTML(filename=str(HTML_IN), base_url=str(PAPER_DIR)).write_pdf(
        target=str(PDF_OUT),
        stylesheets=[CSS(filename=str(CSS_IN))],
    )
    size_kb = PDF_OUT.stat().st_size / 1024
    print(f"Wrote {PDF_OUT}  ({size_kb:.1f} KB)")
    print(f"Build date: {date.today().isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
