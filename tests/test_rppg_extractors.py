"""Tests for the classical rPPG extractors.

Strategy: synthesize a mean-RGB time-series where the green channel carries a
known sinusoidal pulse at a target heart rate. Run CHROM and POS on it. The
peak of the resulting pulse spectrum should land within +/- 3 bpm of the truth.

This validates the extractor math without needing real face video.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import welch

from rppg_sa.extractors.chrom import chrom
from rppg_sa.extractors.pos import pos


def _synthesize_rgb(
    duration_s: float = 30.0,
    fs: float = 30.0,
    hr_bpm: float = 72.0,
    pulse_amplitude: float = 0.02,
    noise_std: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Build a (T, 3) mean-RGB trace with a known pulse in the green channel.

    Skin reflectance model (de Haan & Jeanne 2013 inspiration): the pulse
    modulates green more than red/blue. We add a slow DC drift and white noise
    to mimic motion / illumination artefacts.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_s * fs)) / fs
    f_pulse = hr_bpm / 60.0

    # Skin baseline (typical mid-tone, BGR-agnostic since we returned RGB).
    base = np.array([170.0, 130.0, 110.0])
    # Pulse imprints differently on each channel; green dominates.
    pulse = np.sin(2.0 * np.pi * f_pulse * t)
    channel_weights = np.array([0.35, 1.00, 0.25])  # R, G, B sensitivity to pulse
    pulse_signal = base[None, :] * (1.0 + pulse_amplitude * pulse[:, None] * channel_weights[None, :])

    # Slow lighting drift (0.05 Hz) and white noise.
    drift = 0.005 * np.sin(2.0 * np.pi * 0.05 * t)[:, None] * base[None, :]
    noise = noise_std * rng.standard_normal(pulse_signal.shape) * base[None, :]

    return pulse_signal + drift + noise


def _peak_hr_bpm(pulse: np.ndarray, fs: float, hr_band: tuple[float, float] = (0.7, 4.0)) -> float:
    """Locate the dominant peak in the HR band and return its frequency in bpm."""
    f, pxx = welch(pulse, fs=fs, nperseg=min(len(pulse), 512))
    band = (f >= hr_band[0]) & (f <= hr_band[1])
    f_band = f[band]
    p_band = pxx[band]
    return float(f_band[int(np.argmax(p_band))] * 60.0)


@pytest.mark.parametrize("hr_bpm", [55.0, 72.0, 95.0, 120.0])
def test_chrom_recovers_known_hr(hr_bpm: float) -> None:
    fs = 30.0
    rgb = _synthesize_rgb(duration_s=30.0, fs=fs, hr_bpm=hr_bpm)
    pulse = chrom(rgb, fs=fs)
    hr_hat = _peak_hr_bpm(pulse, fs=fs)
    assert abs(hr_hat - hr_bpm) <= 3.0, f"CHROM HR error too large: got {hr_hat:.1f} vs truth {hr_bpm:.1f}"


@pytest.mark.parametrize("hr_bpm", [55.0, 72.0, 95.0, 120.0])
def test_pos_recovers_known_hr(hr_bpm: float) -> None:
    fs = 30.0
    rgb = _synthesize_rgb(duration_s=30.0, fs=fs, hr_bpm=hr_bpm)
    pulse = pos(rgb, fs=fs)
    hr_hat = _peak_hr_bpm(pulse, fs=fs)
    assert abs(hr_hat - hr_bpm) <= 3.0, f"POS HR error too large: got {hr_hat:.1f} vs truth {hr_bpm:.1f}"


def test_chrom_input_shape_validation() -> None:
    with pytest.raises(ValueError):
        chrom(np.zeros((100,)), fs=30.0)
    with pytest.raises(ValueError):
        chrom(np.zeros((100, 4)), fs=30.0)


def test_pos_input_shape_validation() -> None:
    with pytest.raises(ValueError):
        pos(np.zeros((100,)), fs=30.0)
    with pytest.raises(ValueError):
        pos(np.zeros((100, 4)), fs=30.0)
