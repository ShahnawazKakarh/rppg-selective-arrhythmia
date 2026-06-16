"""Split Conformal Prediction (Angelopoulos & Bates, 2021).

Distribution-free finite-sample coverage guarantee: with held-out calibration
data, the returned prediction sets contain the true label with probability at
least 1 - alpha, marginally over the test distribution (under exchangeability).

Used here both for set-valued predictions and to derive a continuous
"conformity score" that feeds the selective head.
"""
from __future__ import annotations

import numpy as np


def calibrate_threshold(
    cal_probs: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Compute the conformal threshold q-hat from calibration data.

    Uses the standard non-conformity score s_i = 1 - p_i[y_i] (softmax score
    of the true class).

    Args:
        cal_probs: (N_cal, K) calibration-set softmax probabilities.
        cal_labels: (N_cal,) integer ground-truth labels.
        alpha: Miscoverage level. Sets cover the true label with probability
            >= 1 - alpha.

    Returns:
        Scalar threshold q-hat. Include all classes with score s <= q-hat.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    n = len(cal_labels)
    scores = 1.0 - cal_probs[np.arange(n), cal_labels]
    # Finite-sample correction: q-level = ceil((n+1)(1-alpha)) / n.
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = float(np.clip(q_level, 0.0, 1.0))
    return float(np.quantile(scores, q_level, method="higher"))


def predict_sets(
    test_probs: np.ndarray,
    q_hat: float,
) -> list[np.ndarray]:
    """Build conformal prediction sets at threshold q-hat.

    Returns:
        List of length N; each element is an int array of class indices.
    """
    scores = 1.0 - test_probs
    in_set = scores <= q_hat
    return [np.flatnonzero(row) for row in in_set]


def set_sizes(sets: list[np.ndarray]) -> np.ndarray:
    """Per-sample prediction-set size. Larger = more uncertain."""
    return np.array([len(s) for s in sets], dtype=np.int64)


def empirical_coverage(sets: list[np.ndarray], labels: np.ndarray) -> float:
    """Fraction of test samples whose true label lies in their prediction set."""
    covered = [label in s for s, label in zip(sets, labels)]
    return float(np.mean(covered))
