"""CHROM rPPG algorithm (de Haan & Jeanne, IEEE TBME 2013).

Recovers a pulse waveform from a chrominance projection of mean-RGB skin
reflection. Standard classical baseline against which learned extractors are
compared.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 4.0, order: int = 4) -> np.ndarray:
    """Zero-phase bandpass filter targeting the physiological HR range [42, 240] bpm."""
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def chrom(rgb: np.ndarray, fs: float, window_seconds: float = 1.6, overlap: float = 0.5) -> np.ndarray:
    """Apply the CHROM algorithm to a mean-RGB time-series.

    Args:
        rgb: (T, 3) array of mean RGB values per frame.
        fs: Frame rate in Hz.
        window_seconds: Sliding-window length for normalization.
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

    # Hanning window mitigates boundary discontinuities between overlapping segments.
    hann = np.hanning(win)

    for start in range(0, T - win + 1, step):
        end = start + win
        seg = rgb[start:end]
        mean_seg = seg.mean(axis=0) + 1e-8
        norm = seg / mean_seg  # temporal normalization per channel

        # CHROM projection vectors (de Haan & Jeanne, 2013).
        x = 3.0 * norm[:, 0] - 2.0 * norm[:, 1]
        y = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]

        # Adaptive scaling to suppress motion-induced specular component.
        alpha = (np.std(x) + 1e-8) / (np.std(y) + 1e-8)
        s = x - alpha * y
        s = s * hann

        pulse[start:end] += s
        counts[start:end] += hann

    counts[counts == 0] = 1.0
    pulse = pulse / counts
    return _bandpass(pulse, fs)
