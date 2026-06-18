"""Heart-rate estimation from a pulse waveform.

The classical rPPG extractors (CHROM, POS) and the contact PPG ground truth
all return a 1-D pulse waveform. This module converts that into a single
heart-rate scalar.

**Single-clip Welch peak vs. windowed median.** Taking the Welch peak over an
entire 3-minute clip looks principled but breaks badly on noisy video-derived
pulses: lighting flicker, motion sway, breathing modulation, or the 2nd
harmonic of the pulse can all dominate the integrated power over a long
window, even when the true HR is clearly present in shorter sub-segments.
Validating on n = 100 MCD-rPPG records made this concrete (`mcd_rppg_v1`):
POS locked onto 120-155 bpm or 47-65 bpm on more than half of clips, while
the matching PPG-sync GT recovered the true HR from the *same* spectral
analysis. The fix is to estimate HR in shorter windows (10 s is standard in
the rPPG literature) and aggregate by **median**, which is robust to a few
bad windows.

This is also the natural per-segment HR estimator required by the selective-
prediction layer downstream: each window can carry its own confidence, and
the selective head can defer on a per-window basis.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


# Adult HR range, with a small margin. 42-180 bpm covers resting through
# post-exercise; tighter than the 0.7-4 Hz used for the bandpass filter so we
# don't lock onto high-frequency harmonics or low-frequency sway.
DEFAULT_HR_BAND_HZ: tuple[float, float] = (0.7, 3.0)


def peak_hr_bpm(
    pulse: np.ndarray,
    fs: float,
    hr_band: tuple[float, float] = DEFAULT_HR_BAND_HZ,
    nperseg: int | None = None,
) -> float:
    """Spectral peak HR in bpm within `hr_band`.

    Args:
        pulse: 1-D pulse waveform.
        fs: Sampling rate in Hz.
        hr_band: (low, high) Hz, the physiologically plausible range.
        nperseg: Welch window length in samples. Defaults to min(len(pulse), 1024).

    Returns:
        Peak frequency in bpm, or NaN if the band is empty for this fs.
    """
    if len(pulse) < 8:
        return float("nan")
    if nperseg is None:
        nperseg = min(len(pulse), 1024)
    f, pxx = welch(pulse, fs=fs, nperseg=nperseg)
    band = (f >= hr_band[0]) & (f <= hr_band[1])
    if not band.any():
        return float("nan")
    f_band, p_band = f[band], pxx[band]
    return float(f_band[int(np.argmax(p_band))] * 60.0)


def windowed_hr_bpm(
    pulse: np.ndarray,
    fs: float,
    window_seconds: float = 10.0,
    overlap: float = 0.5,
    hr_band: tuple[float, float] = DEFAULT_HR_BAND_HZ,
) -> dict:
    """Estimate HR per sliding window and aggregate by median.

    Args:
        pulse: 1-D pulse waveform.
        fs: Sampling rate in Hz.
        window_seconds: Window length for per-segment HR. 10 s is standard.
        overlap: Fractional overlap between consecutive windows.
        hr_band: (low, high) Hz.

    Returns:
        dict with keys:
          hr_bpm: median HR across windows (NaN if no valid windows).
          hr_bpm_per_window: list of per-window HR values.
          hr_std_bpm: std across windows (HR variability proxy).
          n_windows: number of valid windows.
    """
    win = max(int(round(window_seconds * fs)), 16)
    step = max(int(round(win * (1.0 - overlap))), 1)
    T = len(pulse)
    if T < win:
        # Fall back to a single estimate over the whole clip.
        hr = peak_hr_bpm(pulse, fs, hr_band=hr_band)
        return {
            "hr_bpm": hr,
            "hr_bpm_per_window": [hr],
            "hr_std_bpm": 0.0,
            "n_windows": 1,
        }

    per_window: list[float] = []
    # nperseg for the per-window FFT \u2014 use the window itself so freq resolution
    # is consistent across windows.
    nperseg = win
    for start in range(0, T - win + 1, step):
        seg = pulse[start : start + win]
        seg = seg - seg.mean()
        hr = peak_hr_bpm(seg, fs, hr_band=hr_band, nperseg=nperseg)
        if not np.isnan(hr):
            per_window.append(hr)

    if not per_window:
        return {
            "hr_bpm": float("nan"),
            "hr_bpm_per_window": [],
            "hr_std_bpm": float("nan"),
            "n_windows": 0,
        }

    arr = np.asarray(per_window)
    return {
        "hr_bpm": float(np.median(arr)),
        "hr_bpm_per_window": [float(x) for x in arr],
        "hr_std_bpm": float(np.std(arr)),
        "n_windows": len(arr),
    }
