"""Room-reverb + noise/music augmentation for positive clips (milestone M4).

The #2 data-quality lever: synthetic positives recorded "in a clean studio" don't match
a wake word spoken across a room with a TV on. We convolve each positive with a random
room impulse response (RIR) and mix in background noise/music at a sampled signal-to-noise
ratio, so the trainer sees far-field, noisy variants of every positive.

Pure NumPy on purpose (no scipy/audiomentations) so the mixing math is deterministic and
unit-testable; ``soundfile`` I/O lives in the ``augment.py`` script, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


def rms(signal: np.ndarray) -> float:
    """Root-mean-square level of a signal (0.0 for empty/silent)."""
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal.astype(np.float64)))))


def fit_noise(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Return exactly ``length`` samples of ``noise`` (random offset; tiled if short)."""
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if noise.size == 0:
        return np.zeros(length, dtype=np.float32)
    if noise.size < length:
        reps = int(np.ceil(length / noise.size))
        noise = np.tile(noise, reps)
    max_offset = noise.size - length
    offset = int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
    return noise[offset : offset + length].astype(np.float32, copy=False)


def mix_at_snr(
    signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Mix ``noise`` into ``signal`` at a target ``snr_db`` (dB), peak-safe.

    Noise is length-matched to the signal, then scaled so
    ``10*log10(P_signal / P_noise) == snr_db``. The result is scaled down if it would
    clip [-1, 1]. A silent signal is returned unchanged.
    """
    signal = signal.astype(np.float32, copy=False)
    sig_rms = rms(signal)
    if sig_rms < _EPS:
        return signal
    noise = fit_noise(noise, signal.shape[0], rng)
    noise_rms = rms(noise)
    if noise_rms < _EPS:
        return signal

    target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    scaled = noise * (target_noise_rms / noise_rms)
    mixed = signal + scaled

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32, copy=False)


def apply_rir(signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve ``signal`` with a room impulse response, preserving length + level.

    The RIR is normalized and aligned to its peak (direct-path) tap so the convolved
    output stays time-aligned with the input and the same length.
    """
    signal = signal.astype(np.float32, copy=False)
    if rir.size == 0 or signal.size == 0:
        return signal
    rir = rir.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(rir)))
    if peak < _EPS:
        return signal
    rir = rir / peak
    direct = int(np.argmax(np.abs(rir)))
    convolved = np.convolve(signal, rir)[direct : direct + signal.shape[0]]

    out_rms = rms(convolved)
    in_rms = rms(signal)
    if out_rms > _EPS:  # restore original loudness
        convolved = convolved * (in_rms / out_rms)
    out_peak = float(np.max(np.abs(convolved))) if convolved.size else 0.0
    if out_peak > 1.0:
        convolved = convolved / out_peak
    return convolved.astype(np.float32, copy=False)


@dataclass(frozen=True)
class AugmentStep:
    """One augmented output: which round, its SNR, and whether reverb is applied."""

    round_index: int
    snr_db: float
    use_rir: bool


def build_augmentation_plan(
    rounds: int,
    *,
    snr_range: tuple[float, float] = (5.0, 20.0),
    rir_probability: float = 0.75,
    seed: int = 1337,
) -> list[AugmentStep]:
    """Deterministically plan ``rounds`` augmentations per clip.

    Each round samples an SNR uniformly from ``snr_range`` and applies reverb with
    probability ``rir_probability``. Seeded so a run reproduces exactly.
    """
    rng = np.random.default_rng(seed)
    lo, hi = snr_range
    plan: list[AugmentStep] = []
    for r in range(max(rounds, 0)):
        plan.append(
            AugmentStep(
                round_index=r,
                snr_db=float(rng.uniform(lo, hi)),
                use_rir=bool(rng.random() < rir_probability),
            )
        )
    return plan
