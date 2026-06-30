"""Generate IEEE-grade paper figures (v2) for the v1.9.0 manuscript.

Adds the figures missing from the v1 figure set so the paper hits the
visual quality bar expected of IEEE Access / IEEE JBHI submissions.

Produces:
    paper/figures/confusion_matrices_grid.{pdf,png}   - 2x3 grid, 6 UQ configs
    paper/figures/reliability_diagrams_grid.{pdf,png} - 2x3 grid, calibration
    paper/figures/risk_coverage_overlay.{pdf,png}     - all methods on one plot
    paper/figures/lw_ccsd_margin_bar.{pdf,png}        - delta-AURC per method
    paper/figures/kl_annealing_sensitivity.{pdf,png}  - KL ∈ {1,5,10,15,25}

All figures generated from existing prediction CSVs in runs/<run>/eval_*/.
No model retraining required. Vector PDF + high-DPI PNG.

Usage:
    python scripts/generate_paper_figures_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
FIGURES = REPO_ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Configs — (display name, predictions.csv path, color)
CONFIGS = [
    ("Deterministic",
     RUNS / "synth_rppg_cinc/eval_conformal/predictions.csv",
     "#4477AA"),
    ("MC Dropout (T=30)",
     RUNS / "synth_rppg_cinc/eval_mc_dropout/predictions.csv",
     "#EE6677"),
    ("Clean Ensemble (M=5)",
     RUNS / "synth_rppg_cinc_clean_ens1/eval_ensembles/predictions.csv",
     "#228833"),
    ("SNGP",
     RUNS / "synth_rppg_cinc_sngp/eval_sngp/predictions.csv",
     "#CCBB44"),
    ("EDL",
     RUNS / "synth_rppg_cinc_edl/eval_evidential/predictions.csv",
     "#66CCEE"),
    ("EDL Ensemble (M=5)",
     RUNS / "synth_rppg_cinc_edl/eval_evidential_ensemble/predictions.csv",
     "#AA3377"),
]

LABEL_NAMES = ["NSR", "AF", "Other"]


# IEEE-aligned matplotlib defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
})


def _load_predictions(path: Path) -> pd.DataFrame | None:
    """Defensive loader — return None if the file is missing."""
    if not path.exists():
        print(f"  WARN: {path.relative_to(REPO_ROOT)} not found, skipping")
        return None
    return pd.read_csv(path)


# ============================================================================
# Figure: Confusion matrices grid (2x3)
# ============================================================================
def fig_confusion_matrices() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("ieee_green", ["#f1faee", "#1d3557"])

    for ax, (name, path, _color) in zip(axes.flat, CONFIGS):
        df = _load_predictions(path)
        if df is None:
            ax.axis("off")
            ax.set_title(f"{name}\n(missing)", fontsize=9)
            continue

        cm = confusion_matrix(df["label"], df["pred"], labels=[0, 1, 2])
        # Annotate raw counts
        im = ax.imshow(cm, cmap=cmap, aspect="equal")
        for i in range(3):
            for j in range(3):
                txt_color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=txt_color, fontsize=8, fontweight="bold")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(LABEL_NAMES, fontsize=8)
        ax.set_yticklabels(LABEL_NAMES, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name, fontsize=9, pad=4)
        # Restore spines for matrix
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    fig.suptitle("Confusion matrices across UQ configurations (CinC test, n=6,750)",
                 fontsize=10, y=1.02)
    for ext in ("pdf", "png"):
        out = FIGURES / f"confusion_matrices_grid.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


# ============================================================================
# Figure: Reliability diagrams grid (2x3)
# ============================================================================
def fig_reliability_diagrams() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), constrained_layout=True)

    for ax, (name, path, color) in zip(axes.flat, CONFIGS):
        df = _load_predictions(path)
        if df is None:
            ax.axis("off")
            ax.set_title(f"{name}\n(missing)", fontsize=9)
            continue

        # Max-class probability vs accuracy
        probs = df[[c for c in df.columns if c.startswith("p_")]].to_numpy()
        if probs.shape[1] == 0:
            # Fallback: derive from pred + label
            ax.text(0.5, 0.5, "no per-class probs",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        max_prob = probs.max(axis=1)
        correct = (df["pred"] == df["label"]).astype(int).to_numpy()
        prob_true, prob_pred = calibration_curve(
            correct, max_prob, n_bins=15, strategy="quantile",
        )
        # Compute ECE
        bins = np.linspace(0, 1, 16)
        bin_idx = np.digitize(max_prob, bins) - 1
        ece = 0.0
        for b in range(15):
            mask = bin_idx == b
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_conf = max_prob[mask].mean()
                ece += mask.sum() / len(correct) * abs(bin_acc - bin_conf)

        ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=0.8, label="perfect")
        ax.plot(prob_pred, prob_true, "o-", color=color, linewidth=1.2,
                markersize=4, label=name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{name}\nECE = {ece:.3f}", fontsize=9, pad=4)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linewidth=0.4)

    fig.suptitle("Reliability diagrams across UQ configurations",
                 fontsize=10, y=1.02)
    for ext in ("pdf", "png"):
        out = FIGURES / f"reliability_diagrams_grid.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


# ============================================================================
# Figure: Risk-coverage curves overlay
# ============================================================================
def fig_risk_coverage_overlay() -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)

    for name, path, color in CONFIGS:
        df = _load_predictions(path)
        if df is None:
            continue
        # Compute risk-coverage curve from confidences and correctness
        prob_cols = [c for c in df.columns if c.startswith("p_")]
        if not prob_cols:
            continue
        conf = df[prob_cols].to_numpy().max(axis=1)
        correct = (df["pred"] == df["label"]).astype(int).to_numpy()
        order = np.argsort(-conf)  # sort by confidence descending
        correct_sorted = correct[order]
        coverages = np.arange(1, len(correct_sorted) + 1) / len(correct_sorted)
        cum_correct = np.cumsum(correct_sorted)
        sel_acc = cum_correct / np.arange(1, len(correct_sorted) + 1)
        risk = 1.0 - sel_acc
        ax.plot(coverages, risk, color=color, linewidth=1.4, label=name)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (1 − accuracy)")
    ax.set_title("Risk–coverage curves (CinC test, n=6,750)", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(loc="upper left", framealpha=0.95)

    for ext in ("pdf", "png"):
        out = FIGURES / f"risk_coverage_overlay.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


# ============================================================================
# Figure: LW-CCSD margin bar chart (hard-coded headline numbers from findings)
# ============================================================================
def fig_lw_ccsd_margin_bar() -> None:
    methods = ["Deterministic", "MC Dropout", "Clean Ensemble", "SNGP", "EDL", "EDL Ensemble"]
    uq_aurc      = [0.2367, 0.1980, 0.2066, 0.2044, 0.1725, 0.1640]
    lwccsd_aurc  = [0.1998, 0.1899, 0.1928, 0.1969, 0.1764, 0.1679]
    delta_pct    = [(u - l) / u * 100 for u, l in zip(uq_aurc, lwccsd_aurc)]
    colors = [cfg[2] for cfg in CONFIGS]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    x = np.arange(len(methods))
    width = 0.38

    # Left: paired UQ-only vs LW-CCSD AURC
    ax = axes[0]
    bars1 = ax.bar(x - width/2, uq_aurc, width, label="UQ-only",
                   color="#bbbbbb", edgecolor="#444", linewidth=0.5)
    bars2 = ax.bar(x + width/2, lwccsd_aurc, width, label="+ LW-CCSD",
                   color=colors, edgecolor="#444", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Test AURC ↓")
    ax.set_title("UQ-only vs LW-CCSD-augmented AURC", fontsize=10)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    ax.set_ylim(0, max(uq_aurc) * 1.15)

    # Right: relative delta with sign
    ax = axes[1]
    bar_colors = ["#228833" if d >= 0 else "#CC3311" for d in delta_pct]
    ax.bar(x, delta_pct, color=bar_colors, edgecolor="#444", linewidth=0.5)
    ax.axhline(0, color="#444", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Δ AURC (%, positive = LW-CCSD helps)")
    ax.set_title("LW-CCSD margin per UQ source", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    for i, d in enumerate(delta_pct):
        ax.text(i, d + (0.5 if d >= 0 else -1.2), f"{d:+.1f}%",
                ha="center", fontsize=7, fontweight="bold")

    for ext in ("pdf", "png"):
        out = FIGURES / f"lw_ccsd_margin_bar.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


# ============================================================================
# Figure: KL annealing sensitivity (hard-coded from Section 5.16 findings)
# ============================================================================
def fig_kl_annealing_sensitivity() -> None:
    # KL annealing epochs (1 = effectively no annealing / KL→∞)
    kl_steps        = [1, 5, 10, 15, 25]
    uq_aurc         = [0.1718, 0.1680, 0.1725, 0.1713, 0.1736]
    lwccsd_aurc     = [0.1703, 0.1709, 0.1764, 0.1748, 0.1788]
    delta_pct       = [(u - l) / u * 100 for u, l in zip(uq_aurc, lwccsd_aurc)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)

    # Left: AURC bars
    ax = axes[0]
    x = np.arange(len(kl_steps))
    width = 0.38
    ax.bar(x - width/2, uq_aurc, width, label="UQ-only",
           color="#66CCEE", edgecolor="#444", linewidth=0.5)
    ax.bar(x + width/2, lwccsd_aurc, width, label="+ LW-CCSD",
           color="#0077BB", edgecolor="#444", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"KL=1\n(no anneal)" if k == 1 else f"KL={k}"
                        for k in kl_steps])
    ax.set_ylabel("Test AURC ↓")
    ax.set_title("EDL AURC across KL annealing schedules", fontsize=10)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    ax.set_ylim(0.15, 0.19)

    # Right: ΔAURC sign — KL=1 should be near zero / positive, others negative
    ax = axes[1]
    bar_colors = ["#228833" if d >= 0 else "#CC3311" for d in delta_pct]
    bars = ax.bar(x, delta_pct, color=bar_colors, edgecolor="#444", linewidth=0.5)
    # Highlight the KL→∞ ablation
    bars[0].set_edgecolor("#000")
    bars[0].set_linewidth(1.5)
    ax.axhline(0, color="#444", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"KL=1\n(KL→∞)" if k == 1 else f"KL={k}"
                        for k in kl_steps])
    ax.set_ylabel("Δ AURC (%, positive = LW-CCSD helps)")
    ax.set_title("LW-CCSD margin vs KL annealing schedule", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    for i, d in enumerate(delta_pct):
        ax.text(i, d + (0.2 if d >= 0 else -0.3), f"{d:+.2f}%",
                ha="center", fontsize=7, fontweight="bold")
    # Annotate the refutation
    ax.annotate("KL→∞ ablation:\nmargin disappears",
                xy=(0, delta_pct[0]), xytext=(0.7, 1.5),
                fontsize=7, ha="left",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.6))

    for ext in ("pdf", "png"):
        out = FIGURES / f"kl_annealing_sensitivity.{ext}"
        fig.savefig(out)
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


def main() -> int:
    print("=== Generating IEEE-grade v2 figures ===\n")
    print("Confusion matrices grid:")
    fig_confusion_matrices()
    print("\nReliability diagrams grid:")
    fig_reliability_diagrams()
    print("\nRisk-coverage overlay:")
    fig_risk_coverage_overlay()
    print("\nLW-CCSD margin bar chart:")
    fig_lw_ccsd_margin_bar()
    print("\nKL annealing sensitivity:")
    fig_kl_annealing_sensitivity()
    print("\n=== Done — check paper/figures/ ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
