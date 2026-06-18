"""Synthesize rPPG-style pulse waveforms from MIT-BIH ECG records.

**Why.** Paired face-video + AF rhythm data is scarce (OBF gated, MAHNOB-HCI
gated, no fully-open AF-labelled rPPG corpus). MIT-BIH has real AF rhythm in
ECG. We convert each ECG R-R interval into a PPG beat, placed with a realistic
PTT lag, downsampled to camera frame rate, and corrupted with rPPG-typical
noise (Gaussian + baseline wander + optional motion bursts and lighting flicker).

This is a **controlled experiment for the rhythm component** of rPPG-AF
detection: AF's irregular R-R intervals become irregular pulse intervals in
the synthetic rPPG, which is the signature any rhythm classifier should learn.
Morphology details that real rPPG also lacks (dicrotic notch, etc.) are
intentionally omitted — the goal is to study UQ + selective prediction
behaviour under controlled signal-to-noise, not to fool a clinician.

The synthesizer is deterministic given a seed so train/val/test segments are
reproducible across runs.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


# PPG beat template parameters. These are deliberately simple — a single
# skewed-Gaussian per beat captures the dominant systolic upstroke that
# rPPG actually recovers (dicrotic notch is mostly lost in the camera band).
DEFAULT_BEAT_DURATION_S: float = 0.50    # ~120 bpm equivalent; clipped to RR
DEFAULT_PTT_S: float = 0.20              # pulse transit time R-peak -> systolic peak
DEFAULT_TARGET_FS: float = 30.0          # mimic webcam frame rate
DEFAULT_NOISE_SIGMA: float = 0.05        # Gaussian noise std after normalization
DEFAULT_BASELINE_FREQ_HZ: float = 0.15   # respiration-band baseline wander


def beat_template(n_samples: int, skew: float = 0.4) -> np.ndarray:
    """Single PPG beat — asymmetric Gaussian, normalized to unit peak.

    A short rising edge and a longer falling edge approximate the dominant
    rPPG systolic peak. Skew < 0.5 = peak earlier in the beat.
    """
    if n_samples < 4:
        return np.zeros(max(n_samples, 0), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n_samples)
    peak_t = skew
    sigma_left = peak_t / 2.5
    sigma_right = (1.0 - peak_t) / 2.5
    sigma = np.where(t <= peak_t, sigma_left, sigma_right)
    beat = np.exp(-0.5 * ((t - peak_t) / sigma) ** 2)
    return beat.astype(np.float32)


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Wrap neurokit2's Pan-Tompkins-style detector. Returns sample indices."""
    import neurokit2 as nk

    ecg_clean = nk.ecg_clean(ecg, sampling_rate=int(fs))
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=int(fs))
    peaks = info.get("ECG_R_Peaks")
    if peaks is None:
        return np.array([], dtype=np.int64)
    return np.asarray(peaks, dtype=np.int64)


def synth_ppg_from_ecg(
    ecg: np.ndarray,
    fs_ecg: float,
    fs_out: float = DEFAULT_TARGET_FS,
    ptt_s: float = DEFAULT_PTT_S,
    noise_sigma: float = DEFAULT_NOISE_SIGMA,
    baseline_freq_hz: float = DEFAULT_BASELINE_FREQ_HZ,
    motion_burst_prob: float = 0.0,
    lighting_flicker_amp: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesize an rPPG-style pulse from an ECG signal.

    Args:
        ecg: 1-D ECG signal at fs_ecg.
        fs_ecg: ECG sampling rate (Hz). 360 for MIT-BIH.
        fs_out: Target rPPG sampling rate (Hz). 30 mimics webcam fps.
        ptt_s: Pulse transit time R-peak → systolic peak (seconds).
        noise_sigma: Gaussian noise std on the normalized output.
        baseline_freq_hz: Low-frequency sinusoidal baseline wander.
        motion_burst_prob: Probability of a motion burst per second of signal.
        lighting_flicker_amp: Amplitude of a 4-Hz cosine flicker (0 = off).
        rng: Optional numpy Generator for reproducibility.

    Returns:
        1-D pulse waveform at fs_out, zero-mean unit-std.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) R-peaks at ECG rate.
    peaks = detect_r_peaks(ecg, fs_ecg)
    if len(peaks) < 2:
        # No peaks — return noise so downstream can still produce a fixed shape.
        n_out = int(round(len(ecg) * fs_out / fs_ecg))
        return rng.normal(0.0, 1.0, size=n_out).astype(np.float32)

    # 2) Build the PPG at ECG rate first, then downsample. This keeps R-peak
    #    placement crisp.
    ppg = np.zeros(len(ecg), dtype=np.float32)
    ptt_samples = int(round(ptt_s * fs_ecg))
    rr_samples = np.diff(peaks)

    for i, peak in enumerate(peaks[:-1]):
        rr = rr_samples[i]
        beat_dur_samples = min(rr, int(round(DEFAULT_BEAT_DURATION_S * fs_ecg)))
        beat_dur_samples = max(beat_dur_samples, 8)
        beat = beat_template(beat_dur_samples)
        # Slight amplitude jitter so beats aren't identical.
        amp = 1.0 + 0.05 * rng.standard_normal()
        start = peak + ptt_samples
        end = start + beat_dur_samples
        if end <= len(ppg):
            ppg[start:end] += amp * beat

    # 3) Baseline wander (respiration) + Gaussian noise + optional artefacts.
    t = np.arange(len(ppg)) / fs_ecg
    ppg = ppg + 0.15 * np.sin(2.0 * np.pi * baseline_freq_hz * t).astype(np.float32)

    if lighting_flicker_amp > 0:
        # 4 Hz is a common LED 50Hz/12.5 ratio after fps aliasing; here we just
        # pick a fixed low-freq flicker line.
        ppg = ppg + lighting_flicker_amp * np.cos(2.0 * np.pi * 4.0 * t).astype(np.float32)

    if motion_burst_prob > 0:
        duration_s = len(ecg) / fs_ecg
        n_bursts = rng.poisson(motion_burst_prob * duration_s)
        for _ in range(int(n_bursts)):
            center = int(rng.uniform(0, len(ppg)))
            width = int(0.5 * fs_ecg)  # 0.5s burst
            start = max(0, center - width // 2)
            end = min(len(ppg), center + width // 2)
            ppg[start:end] += rng.normal(0.0, 0.5, size=end - start).astype(np.float32)

    # 4) Downsample to camera frame rate.
    from fractions import Fraction

    ratio = Fraction(fs_out).limit_denominator(1000) / Fraction(fs_ecg).limit_denominator(1000)
    ppg = resample_poly(ppg, up=ratio.numerator, down=ratio.denominator)
    ppg = ppg.astype(np.float32)

    # 5) Add Gaussian noise post-downsample (mimics quantization + camera noise).
    ppg = ppg + noise_sigma * rng.standard_normal(len(ppg)).astype(np.float32)

    # 6) Normalize.
    ppg = ppg - ppg.mean()
    ppg = ppg / (ppg.std() + 1e-8)
    return ppg
