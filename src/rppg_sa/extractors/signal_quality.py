"""Signal-quality estimates for rPPG pulse waveforms.

Used both as a diagnostic (per-segment QC) and as an input to the
signal-quality-aware deferral policy in the selective-prediction head.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def spectral_snr(pulse: np.ndarray, fs: float, hr_band: tuple[float, float] = (0.7, 4.0)) -> float:
    """SNR of the dominant peak in the physiological HR band vs. surrounding noise.

    A higher value indicates a cleaner pulse signal.

    Args:
        pulse: 1-D pulse waveform.
        fs: Sampling rate in Hz.
        hr_band: (low, high) Hz band defining the physiological HR range.

    Returns:
        SNR in dB.
    """
    f, pxx = welch(pulse, fs=fs, nperseg=min(len(pulse), 512))
    band = (f >= hr_band[0]) & (f <= hr_band[1])
    if not band.any():
        return float("-inf")

    pxx_band = pxx[band]
    f_band = f[band]
    peak_idx = int(np.argmax(pxx_band))
    peak_f = f_band[peak_idx]
    peak_p = pxx_band[peak_idx]

    # Noise: power in the HR band, excluding a ±0.2 Hz neighborhood of the peak
    # and its first harmonic.
    excl = (np.abs(f_band - peak_f) < 0.2) | (np.abs(f_band - 2 * peak_f) < 0.2)
    noise_p = pxx_band[~excl].mean() + 1e-12
    return 10.0 * float(np.log10(peak_p / noise_p))


def template_sqi(pulse: np.ndarray, fs: float, ibi_min: float = 0.4, ibi_max: float = 1.5) -> float:
    """Template-matching signal-quality index.

    Segments the pulse into beat-length windows around detected peaks, builds a
    median template, and returns the mean correlation of each beat with the
    template. Values closer to 1.0 indicate consistent beat morphology.

    Args:
        pulse: 1-D pulse waveform (zero-mean recommended).
        fs: Sampling rate in Hz.
        ibi_min: Minimum inter-beat interval in seconds.
        ibi_max: Maximum inter-beat interval in seconds.

    Returns:
        Mean per-beat correlation in [-1, 1]. Returns 0.0 if no beats found.
    """
    from scipy.signal import find_peaks

    distance = int(round(ibi_min * fs))
    peaks, _ = find_peaks(pulse, distance=distance)
    if len(peaks) < 3:
        return 0.0

    # Use median IBI to set a symmetric beat window.
    ibis = np.diff(peaks) / fs
    median_ibi = float(np.median(ibis))
    median_ibi = float(np.clip(median_ibi, ibi_min, ibi_max))
    half = int(round(median_ibi * fs / 2))
    if half < 4:
        return 0.0

    beats: list[np.ndarray] = []
    for p in peaks:
        if p - half < 0 or p + half >= len(pulse):
            continue
        beats.append(pulse[p - half : p + half])
    if len(beats) < 3:
        return 0.0

    beats_arr = np.stack(beats, axis=0)  # (n_beats, 2*half)
    template = np.median(beats_arr, axis=0)
    template = (template - template.mean()) / (template.std() + 1e-8)

    corrs = []
    for b in beats_arr:
        bn = (b - b.mean()) / (b.std() + 1e-8)
        corrs.append(float(np.dot(bn, template) / len(template)))
    return float(np.mean(corrs))


def summarize_quality(pulse: np.ndarray, fs: float) -> dict[str, float]:
    """Convenience: return both SNR and template SQI for a pulse segment."""
    return {
        "snr_db": spectral_snr(pulse, fs),
        "template_sqi": template_sqi(pulse, fs),
    }
