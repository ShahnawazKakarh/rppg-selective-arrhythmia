# Paper draft

`main.tex` — IEEE J-BHI submission draft for **Learned Class-Conditional Signal-Quality Deferral for Selective rPPG-Based Atrial Fibrillation Screening**.

## Status

Skeleton + abstract + introduction + methods equations + result tables stubbed.
Sections marked `[TODO]` need prose.

## Compile

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Requires `texlive-publishers` (IEEEtran class) and `texlive-bibtex-extra`.

## Publication plan

1. **TechRxiv** — no endorsement needed; immediate DOI. IEEE-affiliated, direct fit for J-BHI submission.
2. **Zenodo** — auto-DOI on GitHub release; archives code + paper PDF together.
3. **arXiv (cs.LG or eess.SP)** — request endorsement once TechRxiv DOI is live; cite the author's existing 3 ORCID-indexed DOIs as credentials.
4. **IEEE J-BHI** — preprints are not disqualifying; submit after TechRxiv.

## Outstanding before submission

- Prose for sections marked `[TODO]`: Related Work, Methods, Discussion.
- Pareto-frontier figure (replace Table II with a plot at submission time).
- Cross-UQ replication table.
- Confirm Egorov et al. 2025 MCD-rPPG citation details.
- Decide whether to cite OACSP / retinal-selective-prediction as related self-work.
- Add reproducibility statement subsection if J-BHI now requires one (check current author guidelines).
