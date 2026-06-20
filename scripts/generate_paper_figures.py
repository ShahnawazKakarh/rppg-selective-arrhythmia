"""Generate IEEE-style figures for the paper from existing run artefacts.

Outputs three vector PDFs into paper/figures/:
  1. naive_sqi_failure.pdf   AF recall vs coverage: UQ-only vs naive SQI vs af_immune.
  2. pareto_frontier.pdf     LW-CCSD AURC improvement vs AF recall cost across val floors.
  3. snr_distribution.pdf    Per-class SNR histograms on CinC test (mechanism).

Each figure is single-column-width (3.5 in) by default; pass --width 7.16 for
double-column. Vector PDF; matplotlib style tuned for IEEE journal.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from rppg_sa.data.cinc2017_synth_torch import (
    CinCSynthRPPGSegmentDataset,
    subject_disjoint_split,
)
from rppg_sa.extractors.signal_quality import summarize_quality
from rppg_sa.utils.config import load_config


# IEEE-friendly style: serif, modest font sizes, thin lines.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ----------------------------------------------------------------------
# Figure 1 - Naive SQI failure: AF recall vs coverage by policy
# ----------------------------------------------------------------------
def fig_naive_sqi_failure(out: Path, width: float) -> None:
    # Numbers taken directly from per_class_sqi_deferral and
    # class_conditional_sqi outputs on synth_rppg_cinc deterministic run.
    cov = np.array([0.50, 0.70, 0.80, 0.90, 0.95, 1.00])
    af_uq      = np.array([0.707, 0.732, 0.764, 0.800, 0.814, 0.828])
    af_naive   = np.array([0.043, 0.365, 0.637, 0.730, 0.778, 0.828])
    af_immune  = np.array([0.689, 0.717, 0.732, 0.767, 0.798, 0.828])
    af_lw      = np.array([0.626, 0.698, 0.726, 0.773, 0.798, 0.828])  # floor 0.55

    fig, ax = plt.subplots(figsize=(width, width * 0.62))
    ax.plot(cov, af_uq,     marker="o", label="UQ-only",                 color="#1f77b4")
    ax.plot(cov, af_naive,  marker="s", label=r"Na\"ive SQI ($w{=}0.70$)",   color="#d62728", linestyle="--")
    ax.plot(cov, af_immune, marker="^", label=r"AF-immune ($w{=}0.70$)",      color="#9467bd", linestyle=":")
    ax.plot(cov, af_lw,     marker="D", label=r"LW-CCSD (floor $0.55$)",     color="#2ca02c")

    ax.set_xlabel("Coverage")
    ax.set_ylabel("AF recall on kept subset")
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.0, 0.95)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95)
    # Annotation: highlight the collapse.
    ax.annotate(
        r"AF recall $0.71\!\to\!0.04$",
        xy=(0.50, 0.043), xytext=(0.55, 0.27),
        arrowprops=dict(arrowstyle="->", lw=0.5, color="#d62728"),
        fontsize=7, color="#d62728",
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), format="png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out} (+ .png)")


# ----------------------------------------------------------------------
# Figure 2 - LW-CCSD Pareto frontier
# ----------------------------------------------------------------------
def fig_pareto_frontier(out: Path, width: float) -> None:
    # (Val AF floor, test AURC, AF cost) from optimize_class_conditional_weights runs.
    floors    = [0.60, 0.58, 0.55, 0.50, 0.45, 0.40]
    test_aurc = [0.2270, 0.2082, 0.1998, 0.2020, 0.1984, 0.1958]
    af_cost   = [-0.004, -0.045, -0.081, -0.123, -0.166, -0.220]
    uq_aurc        = 0.2367
    naive_aurc     = 0.1942
    naive_af_cost  = -0.664

    fig, ax = plt.subplots(figsize=(width, width * 0.62))
    # Pareto curve (lower-left is better: low AURC, small AF cost magnitude)
    ax.plot([abs(c) for c in af_cost], test_aurc,
            marker="o", color="#2ca02c", label="LW-CCSD (varying floor)")
    # Reference points
    ax.scatter([0.0], [uq_aurc],    marker="s", s=60, color="#1f77b4", zorder=5,
               label="UQ-only baseline")
    ax.scatter([abs(naive_af_cost)], [naive_aurc], marker="X", s=70, color="#d62728", zorder=5,
               label=r"Na\"ive SQI ($w{=}0.70$)")

    # Annotate every floor.
    for f, c, a in zip(floors, af_cost, test_aurc):
        ax.annotate(f"floor {f:.2f}", (abs(c), a),
                    textcoords="offset points", xytext=(5, 4), fontsize=6.5,
                    color="#2ca02c")
    # Highlight recommended operating point.
    rec_idx = floors.index(0.55)
    ax.scatter([abs(af_cost[rec_idx])], [test_aurc[rec_idx]],
               marker="o", facecolors="none", edgecolors="#2ca02c",
               s=110, lw=1.2, zorder=4)
    ax.annotate("recommended\n(+15.6\\,\\% AURC,\n-0.081 AF recall)",
                (abs(af_cost[rec_idx]), test_aurc[rec_idx]),
                textcoords="offset points", xytext=(15, -22), fontsize=7,
                ha="left", color="#2ca02c")

    ax.set_xlabel("AF recall cost (absolute, at coverage 0.50)")
    ax.set_ylabel(r"Test AURC ($\downarrow$ better)")
    ax.set_xlim(-0.03, 0.72)
    ax.set_ylim(0.185, 0.245)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), format="png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out} (+ .png)")


# ----------------------------------------------------------------------
# Figure 3 - Per-class SNR distributions on CinC test
# ----------------------------------------------------------------------
def fig_snr_distribution(out: Path, width: float, config_path: Path, run_dir: Path) -> None:
    cfg = load_config(config_path)
    ds = CinCSynthRPPGSegmentDataset(
        root=cfg["data"]["root"],
        target_fs=float(cfg["data"]["target_fs"]),
        window_seconds=float(cfg["data"]["window_seconds"]),
        step_seconds=float(cfg["data"].get("step_seconds", cfg["data"]["window_seconds"])),
        cache_dir=cfg["data"].get("cache_dir"),
        synth_seed=int(cfg["data"].get("synth_seed", 42)),
        noise_sigma=float(cfg["data"].get("noise_sigma", 0.05)),
    )
    splits_file = run_dir / "splits.json"
    with splits_file.open() as f:
        sp = json.load(f)
    test_idx = sp["test_idx"]

    fs = float(cfg["data"]["target_fs"])
    snr_per_class: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for i in test_idx:
        seg = ds.signals[i]
        q = summarize_quality(seg, fs=fs)
        snr_per_class[ds.labels[i]].append(q["snr_db"])

    labels = ["NSR", "AF", "Other"]
    colors = ["#1f77b4", "#d62728", "#7f7f7f"]
    fig, ax = plt.subplots(figsize=(width, width * 0.62))
    bins = np.linspace(-5, 25, 50)
    for c, name, col in zip(range(3), labels, colors):
        vals = np.array(snr_per_class[c])
        ax.hist(vals, bins=bins, density=True, alpha=0.45, label=f"{name}  (n={len(vals)}, med {np.median(vals):.1f}\\,dB)",
                color=col, edgecolor=col, linewidth=0.6)
        ax.axvline(np.median(vals), color=col, linestyle="--", linewidth=0.7, alpha=0.7)

    ax.set_xlabel("Spectral SNR in cardiac band (dB)")
    ax.set_ylabel("Density")
    ax.set_xlim(-5, 25)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), format="png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {out} (+ .png)")
    print(f"  per-class median SNR: " + ", ".join(
        f"{n}={np.median(snr_per_class[c]):.2f} dB" for c, n in zip(range(3), labels)
    ))


# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/synth_rppg_cinc.yaml"))
    p.add_argument("--run-dir", type=Path, default=Path("runs/synth_rppg_cinc"))
    p.add_argument("--out", type=Path, default=Path("paper/figures"))
    p.add_argument("--width", type=float, default=3.5,
                   help="Figure width in inches (3.5 single column, 7.16 double).")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Figure 1: naive SQI AF-recall failure")
    fig_naive_sqi_failure(args.out / "naive_sqi_failure.pdf", args.width)

    print("Figure 2: LW-CCSD Pareto frontier")
    fig_pareto_frontier(args.out / "pareto_frontier.pdf", args.width)

    print("Figure 3: per-class SNR distribution on CinC test")
    fig_snr_distribution(args.out / "snr_distribution.pdf", args.width, args.config, args.run_dir)


if __name__ == "__main__":
    main()
