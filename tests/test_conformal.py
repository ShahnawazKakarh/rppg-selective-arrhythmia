"""Tests for the conformal-prediction implementation."""
from __future__ import annotations

import numpy as np
import pytest

from rppg_sa.uncertainty.conformal import (
    calibrate_threshold,
    empirical_coverage,
    predict_sets,
    set_sizes,
)


def _synthetic_softmax(n: int, k: int, accuracy: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate (probs, labels) where the argmax is correct with given accuracy."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    probs = rng.dirichlet(np.ones(k), size=n)
    # Force the correct class to be argmax for `accuracy` fraction of samples.
    mask = rng.uniform(size=n) < accuracy
    for i in np.flatnonzero(mask):
        order = np.argsort(-probs[i])
        if order[0] != labels[i]:
            # Swap the max entry with the true-label entry.
            top = order[0]
            probs[i, [top, labels[i]]] = probs[i, [labels[i], top]]
    return probs, labels


def test_conformal_marginal_coverage() -> None:
    """Empirical coverage should meet the 1-alpha target on held-out data."""
    n_cal, n_test, k = 1000, 1000, 3
    alpha = 0.1
    cal_probs, cal_labels = _synthetic_softmax(n_cal, k, accuracy=0.7, seed=0)
    test_probs, test_labels = _synthetic_softmax(n_test, k, accuracy=0.7, seed=1)

    q_hat = calibrate_threshold(cal_probs, cal_labels, alpha=alpha)
    sets = predict_sets(test_probs, q_hat)
    coverage = empirical_coverage(sets, test_labels)
    # Marginal coverage guarantee: >= 1 - alpha, with finite-sample slack.
    assert coverage >= (1 - alpha) - 0.03


def test_set_sizes_match_input_count() -> None:
    probs = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.4, 0.2],
            [0.34, 0.33, 0.33],
        ]
    )
    sets = predict_sets(probs, q_hat=0.7)
    sizes = set_sizes(sets)
    assert sizes.shape == (3,)
    # Larger threshold => non-decreasing set sizes per row.
    sets_loose = predict_sets(probs, q_hat=0.9)
    assert (set_sizes(sets_loose) >= sizes).all()


def test_calibrate_alpha_validation() -> None:
    probs, labels = _synthetic_softmax(50, 3, accuracy=0.6, seed=0)
    with pytest.raises(ValueError):
        calibrate_threshold(probs, labels, alpha=0.0)
    with pytest.raises(ValueError):
        calibrate_threshold(probs, labels, alpha=1.0)
