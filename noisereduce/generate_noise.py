"""
Band-limited noise generation using FFT-based synthesis.

This module generates noise constrained to a specific frequency band by:
1. Defining a frequency-domain mask (which bins contain energy)
2. Applying random phases to produce stochastic time-domain signal
3. Enforcing Hermitian symmetry so the inverse FFT yields real-valued output

Reference: https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
"""

import numpy as np


def fftnoise(f):
    """
    Generate time-domain noise from a frequency-domain magnitude spectrum by applying random phases.

    Why: Real-valued time signals have a symmetric spectrum (Hermitian). Given only magnitude
    (or power) per frequency bin, we assign random phases and enforce conjugate symmetry
    so that the inverse FFT produces a real-valued signal with the desired frequency content.

    Parameters
    ----------
    f : array-like
        Frequency-domain magnitude mask. Typically 0s and 1s indicating which bins have energy.
        Length determines output length. Index 0 is DC, middle is Nyquist.

    Returns
    -------
    np.ndarray
        Real-valued time-domain noise signal with magnitude spectrum matching the input mask.
    """
    # Ensure complex dtype so we can store phase-multiplied spectrum
    f = np.array(f, dtype="complex")

    # Number of positive-frequency bins (excluding DC and Nyquist)
    # For real FFT: spectrum is symmetric; only first half is independent
    Np = (len(f) - 1) // 2

    # Generate random phases in [0, 2*pi) and convert to unit complex numbers
    # This randomizes the phase of each positive-frequency component
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)

    # Apply phases to positive-frequency bins (indices 1 to Np)
    # Magnitude is preserved; only phase changes
    f[1 : Np + 1] *= phases

    # Enforce Hermitian symmetry: F(-k) = conj(F(k))
    # Negative frequencies (reverse order, excluding DC) must be conjugate of positive
    # This ensures np.fft.ifft() returns a real-valued time signal
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])

    # Inverse FFT to time domain; .real discards negligible imaginary part from float rounding
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    """
    Generate noise confined to a specific frequency band.

    Creates noise whose energy lies entirely within [min_freq, max_freq], useful for
    testing filters, simulating colored noise, or generating synthetic training data.

    Parameters
    ----------
    min_freq : float
        Lower bound of the frequency band (Hz).
    max_freq : float
        Upper bound of the frequency band (Hz).
    samples : int, optional
        Number of time-domain samples to generate (default: 1024).
    samplerate : float, optional
        Sample rate in Hz (default: 1). Used to map bin indices to frequencies.

    Returns
    -------
    np.ndarray
        Real-valued noise signal of length `samples` with band-limited spectrum.
    """
    # Compute center frequency of each FFT bin (in Hz)
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))

    # Build magnitude mask: 1 for bins in [min_freq, max_freq], 0 elsewhere
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1

    # Apply random phases and invert to time domain via fftnoise
    return fftnoise(f)
