"""POS rPPG algorithm (Wang et al., IEEE TBME 2017).

Plane-Orthogonal-to-Skin projection. Empirically more robust to motion than
CHROM and a standard classical baseline in the rPPG literature.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 4.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def pos(rgb: np.ndarray, fs: float, window_seconds: float = 1.6) -> np.ndarray:
    """Apply the POS algorithm to a mean-RGB time-series.

    Args:
        rgb: (T, 3) array of mean RGB values per frame.
        fs: Frame rate in Hz.
        window_seconds: Sliding-window length for temporal normalization.

    Returns:
        (T,) bandpass-filtered pulse waveform.
    """
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must be (T, 3); got {rgb.shape}")

    T = rgb.shape[0]
    win = max(int(round(window_seconds * fs)), 32)
    pulse = np.zeros(T, dtype=np.float64)

    # Projection matrix from Wang et al. 2017, Eq. 10.
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)

    for n in range(win, T):
        seg = rgb[n - win : n]  # (win, 3)
        mean_seg = seg.mean(axis=0) + 1e-8
        Cn = (seg / mean_seg).T  # (3, win), zero-mean / normalized

        S = P @ Cn  # (2, win)
        alpha = (np.std(S[0]) + 1e-8) / (np.std(S[1]) + 1e-8)
        h = S[0] + alpha * S[1]
        h = h - h.mean()

        # Overlap-add: contribute this window's signal to the running pulse.
        pulse[n - win : n] += h

    return _bandpass(pulse, fs)
