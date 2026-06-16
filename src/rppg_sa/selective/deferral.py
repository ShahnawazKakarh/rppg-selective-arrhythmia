"""Signal-quality-aware deferral policy.

Combines model-derived confidence with a physical signal-quality estimate
(from `extractors.signal_quality`) into a unified score used to decide which
samples to retain at a given target coverage.

This is the technical novelty of the selective head: prior rPPG-AF work treats
either the signal as noisy-but-trusted or the model as confident-but-blind;
combining the two should improve risk-coverage curves over either alone.
"""
from __future__ import annotations

import numpy as np


def combine_score(
    model_confidence: np.ndarray,
    signal_quality: np.ndarray,
    quality_weight: float = 0.3,
) -> np.ndarray:
    """Linear combination of model confidence and (rank-normalized) signal quality.

    Args:
        model_confidence: (N,) array. Higher = more confident.
            Typical choice: max softmax probability, or 1 - normalized entropy.
        signal_quality: (N,) array of SQI values (e.g. SNR dB or template SQI).
            Rank-normalized internally so units don't matter.
        quality_weight: Mixing weight in [0, 1]. 0 = model only; 1 = SQI only.

    Returns:
        (N,) combined score; higher = retain.
    """
    if not 0.0 <= quality_weight <= 1.0:
        raise ValueError(f"quality_weight must be in [0, 1]; got {quality_weight}")

    mc = _rank_normalize(model_confidence)
    sq = _rank_normalize(signal_quality)
    return (1.0 - quality_weight) * mc + quality_weight * sq


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Map values to [0, 1] by rank. Robust to scale differences across signals."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    if len(x) <= 1:
        return np.zeros_like(x)
    return ranks / (len(x) - 1)


def select_at_coverage(
    scores: np.ndarray, target_coverage: float
) -> np.ndarray:
    """Boolean mask selecting the top-`target_coverage` fraction by `scores`."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"target_coverage must be in (0, 1]; got {target_coverage}")
    n = len(scores)
    k = max(1, int(np.ceil(target_coverage * n)))
    threshold_idx = np.argsort(-scores)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[threshold_idx] = True
    return mask
