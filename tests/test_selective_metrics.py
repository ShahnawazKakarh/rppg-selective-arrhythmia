"""Tests for selective-prediction metrics."""
from __future__ import annotations

import numpy as np
import pytest

from rppg_sa.selective.metrics import (
    brier_score,
    expected_calibration_error,
    predictive_entropy,
    risk_coverage_curve,
    selective_accuracy_at_coverage,
)


def test_risk_coverage_perfect_ranking() -> None:
    """When confidence perfectly ranks correctness, risk is monotone non-decreasing."""
    confidences = np.array([0.99, 0.95, 0.80, 0.60, 0.40])
    correct = np.array([1, 1, 1, 0, 0])
    rc = risk_coverage_curve(confidences, correct)

    # First three retained samples are all correct -> risk = 0.
    assert rc.risk[2] == pytest.approx(0.0)
    # Full coverage -> risk = 1 - accuracy = 1 - 3/5 = 0.4.
    assert rc.risk[-1] == pytest.approx(0.4)
    # AURC strictly between 0 (oracle) and 0.4 (random).
    assert 0.0 < rc.aurc < 0.4


def test_selective_accuracy_at_coverage() -> None:
    confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([1, 1, 0, 1, 0])
    # Top 60% of 5 samples = top 3. Two of them correct -> 2/3.
    acc = selective_accuracy_at_coverage(confidences, correct, target_coverage=0.6)
    assert acc == pytest.approx(2.0 / 3.0)


def test_ece_perfect_calibration() -> None:
    """A perfectly-calibrated classifier on a balanced binary task has near-zero ECE."""
    rng = np.random.default_rng(0)
    n = 2000
    p = rng.uniform(0.5, 1.0, size=n)
    # With probability p of being correct, sample labels accordingly.
    preds = rng.integers(0, 2, size=n)
    labels = np.where(rng.uniform(size=n) < p, preds, 1 - preds)
    probs = np.stack([1 - p, p], axis=1)
    probs[preds == 0] = probs[preds == 0][:, ::-1]
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert ece < 0.05


def test_brier_score_bounds() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    labels = np.array([0, 1])
    bs = brier_score(probs, labels, num_classes=3)
    assert 0.0 <= bs <= 2.0


def test_predictive_entropy() -> None:
    # Uniform -> max entropy = log(K).
    K = 3
    probs = np.full((1, K), 1.0 / K)
    h = predictive_entropy(probs)
    assert h[0] == pytest.approx(np.log(K), rel=1e-4)

    # One-hot -> entropy = 0.
    probs = np.array([[1.0, 0.0, 0.0]])
    h = predictive_entropy(probs)
    assert h[0] == pytest.approx(0.0, abs=1e-6)
