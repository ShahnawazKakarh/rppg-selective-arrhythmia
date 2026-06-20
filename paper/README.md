# Paper draft

**Learned Class-Conditional Signal-Quality Deferral for Selective rPPG-Based Atrial Fibrillation Screening.**

## Build the PDF (Python-only, no LaTeX required)

```bash
pip install 'weasyprint>=63'
python scripts/generate_paper_figures.py    # produces .pdf + .png figures
python paper/build.py                       # produces paper/lw-ccsd-rppg-af-v1.0.0.pdf
```

The WeasyPrint pipeline renders `paper/report.html` + `paper/style.css` to `paper/lw-ccsd-rppg-af-v1.0.0.pdf`. No LaTeX installation needed.

## Files

- `report.html` — paper content, full prose, four tables, three figures.
- `style.css` — IEEE-aligned academic style, A4 page geometry, Source Serif body, Inter headings/tables.
- `build.py` — single command that drives WeasyPrint.
- `main.tex` + `refs.bib` — also maintained as a LaTeX source for arXiv submission (which requires LaTeX), built with `pdflatex` or `tectonic` separately if/when arXiv endorsement clears.
- `figures/` — three PDFs (`naive_sqi_failure.pdf`, `pareto_frontier.pdf`, `snr_distribution.pdf`) for the LaTeX path, plus matching PNGs for the WeasyPrint path.

## Publication plan

1. **Build the PDF** with `python paper/build.py` — produces `paper/lw-ccsd-rppg-af-v1.0.0.pdf`.
2. **Zenodo (manual upload)** — https://zenodo.org/uploads/new, attach the PDF and the GitHub URL as related identifier, fill the metadata, publish. Instant DOI.
3. **OSF Preprints** — https://osf.io/preprints/, upload the same PDF, choose subject area Computer Sciences / Medical Sciences. Another DOI.
4. **GitHub Release** — `git tag v1.0.0 && git push origin v1.0.0`, create the release with the PDF attached. If Zenodo GitHub integration is enabled, mints a code+paper DOI automatically.
5. **arXiv** — once endorsement clears (separate process for `eess.SP` or `cs.LG`), submit the LaTeX source. TechRxiv was the originally-planned IEEE-aligned preprint server but is currently migrating; revisit when reopened.
6. **IEEE J-BHI** — preprints are not disqualifying; submit the LaTeX source via Manuscript Central after the preprints are live.

## TechRxiv (when it reopens)

Track migration status at https://www.techrxiv.org. Same PDF, no endorsement needed.

## Why two paper sources

- `report.html` + WeasyPrint is the **build-anywhere** path with no toolchain headaches. Use this for Zenodo/OSF/preprint distribution.
- `main.tex` + LaTeX is the **arXiv / IEEE submission** path because arXiv requires LaTeX source. Use this only when submitting to those venues.

Both encode the same paper. Keep them in sync when editing.
