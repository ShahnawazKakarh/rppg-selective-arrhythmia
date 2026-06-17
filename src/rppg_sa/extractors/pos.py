"""POS rPPG algorithm (Wang et al., IEEE TBME 2017).

Plane-Orthogonal-to-Skin projection. Empirically more robust to motion than
CHROM and a standard classical baseline in the rPPG literature.

Reference:
    W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan,
    "Algorithmic Principles of Remote PPG", IEEE TBME 64(7), 2017.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 4.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def pos(rgb: np.ndarray, fs: float, window_seconds: float = 1.6, overlap: float = 0.5) -> np.ndarray:
    """Apply the POS algorithm to a mean-RGB time-series.

    Args:
        rgb: (T, 3) array of mean RGB values per frame.
        fs: Frame rate in Hz.
        window_seconds: Sliding-window length for temporal normalization.
        overlap: Fractional overlap between successive windows (0..1).

    Returns:
        (T,) bandpass-filtered pulse waveform.
    """
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (T, 3); got {rgb.shape}")

    T = rgb.shape[0]
    win = max(int(round(window_seconds * fs)), 32)
    step = max(int(round(win * (1.0 - overlap))), 1)
    pulse = np.zeros(T, dtype=np.float64)
    counts = np.zeros(T, dtype=np.float64)

    # Projection matrix from Wang et al. 2017, Eq. 10.
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)

    # Hanning window mitigates boundary discontinuities between overlapping segments.
    hann = np.hanning(win)

    for start in range(0, T - win + 1, step):
        end = start + win
        seg = rgb[start:end]  # (win, 3)
        mean_seg = seg.mean(axis=0) + 1e-8
        Cn = (seg / mean_seg).T  # (3, win), temporally normalized per channel

        S = P @ Cn  # (2, win) — two projection components
        # Adaptive scaling, Wang et al. 2017 Eq. 11.
        alpha = (np.std(S[0]) + 1e-8) / (np.std(S[1]) + 1e-8)
        # IMPORTANT: minus sign, not plus (canonical POS combination).
        h = S[0] - alpha * S[1]
        h = h - h.mean()
        h = h * hann

        pulse[start:end] += h
        counts[start:end] += hann

    counts[counts == 0] = 1.0
    pulse = pulse / counts
    return _bandpass(pulse, fs)
