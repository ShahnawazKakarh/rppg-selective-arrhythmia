"""Selective-prediction metrics.

Implements the standard evaluation framework for abstention-capable classifiers:
    - Risk-coverage curve
    - AURC (Area Under the Risk-Coverage curve)
    - Selective accuracy at a target coverage
    - Expected Calibration Error (ECE)
    - Brier score (multi-class)

References:
    Geifman & El-Yaniv (2017). "Selective Classification for Deep Neural Networks."
    Geifman et al. (2018). "Bias-Reduced Uncertainty Estimation for Deep Neural Classifiers."
    Naeini et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskCoverage:
    """Result of a risk-coverage sweep.

    Attributes:
        coverage: Monotone-increasing array of coverage values in (0, 1].
        risk: Selective risk (1 - accuracy on retained samples) at each coverage.
        aurc: Area under the risk-coverage curve (lower is better).
    """

    coverage: np.ndarray
    risk: np.ndarray
    aurc: float


def risk_coverage_curve(
    confidences: np.ndarray,
    correct: np.ndarray,
) -> RiskCoverage:
    """Compute the risk-coverage curve from per-sample confidences and correctness.

    Args:
        confidences: (N,) array of per-sample confidence scores. Higher = more
            confident. For UQ methods that produce uncertainty (e.g. entropy),
            pass `-uncertainty` so that confidence sorts high-first.
        correct: (N,) boolean or 0/1 array indicating whether each prediction
            was correct.

    Returns:
        RiskCoverage container with coverage, risk, and AURC.
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidences.shape != correct.shape:
        raise ValueError(
            f"confidences and correct must have the same shape; "
            f"got {confidences.shape} and {correct.shape}"
        )

    n = len(confidences)
    # Sort by descending confidence: we retain the most confident first.
    order = np.argsort(-confidences)
    correct_sorted = correct[order]

    cumulative_correct = np.cumsum(correct_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    selective_acc = cumulative_correct / k
    risk = 1.0 - selective_acc

    # AURC: trapezoidal integration over coverage.
    # np.trapezoid replaces np.trapz (removed in NumPy 2.0).
    _trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    aurc = float(_trapezoid(risk, coverage))
    return RiskCoverage(coverage=coverage, risk=risk, aurc=aurc)


def selective_accuracy_at_coverage(
    confidences: np.ndarray, correct: np.ndarray, target_coverage: float
) -> float:
    """Selective accuracy when retaining the top `target_coverage` fraction by confidence."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"target_coverage must be in (0, 1]; got {target_coverage}")
    rc = risk_coverage_curve(confidences, correct)
    idx = int(np.searchsorted(rc.coverage, target_coverage))
    idx = min(idx, len(rc.coverage) - 1)
    return float(1.0 - rc.risk[idx])


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error with equal-width bins on max-class probability.

    Args:
        probs: (N, K) array of predicted class probabilities (rows sum to 1).
        labels: (N,) array of integer ground-truth labels in [0, K).
        n_bins: Number of equal-width bins on [0, 1].

    Returns:
        ECE as a single scalar in [0, 1]. Lower is better.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    confs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Right-closed bin to include perfect-confidence predictions in the top bin.
        in_bin = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if not in_bin.any():
            continue
        acc_bin = correct[in_bin].mean()
        conf_bin = confs[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Multi-class Brier score. Lower is better; bounded in [0, 2]."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    onehot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def predictive_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-sample predictive entropy in nats. Higher = more uncertain.

    Args:
        probs: (N, K) array of class probabilities.

    Returns:
        (N,) array of entropies in nats.
    """
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)
