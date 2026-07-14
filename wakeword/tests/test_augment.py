"""Tests for the M4 augmentation math (lib/augment.py) — pure NumPy, deterministic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib import augment as aug  # noqa: E402


def test_rms_basic() -> None:
    assert aug.rms(np.zeros(10)) == 0.0
    assert abs(aug.rms(np.ones(10)) - 1.0) < 1e-6


def test_fit_noise_matches_length() -> None:
    rng = np.random.default_rng(0)
    assert aug.fit_noise(np.arange(3.0), 10, rng).shape == (10,)      # tiled up
    assert aug.fit_noise(np.arange(100.0), 10, rng).shape == (10,)     # cropped down
    assert aug.fit_noise(np.zeros(0), 5, rng).shape == (5,)            # empty -> silence


def test_mix_at_snr_achieves_target() -> None:
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    signal = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    noise = 0.1 * rng.standard_normal(16000).astype(np.float32)

    mixed = aug.mix_at_snr(signal, noise, snr_db=10.0, rng=rng)
    added = mixed - signal
    achieved = 20 * np.log10(aug.rms(signal) / aug.rms(added))
    assert abs(achieved - 10.0) < 0.5


def test_mix_at_snr_silent_signal_unchanged() -> None:
    rng = np.random.default_rng(2)
    silent = np.zeros(1000, dtype=np.float32)
    out = aug.mix_at_snr(silent, np.ones(1000, dtype=np.float32), 10.0, rng)
    assert np.array_equal(out, silent)


def test_apply_rir_delta_is_identity() -> None:
    signal = np.linspace(-0.5, 0.5, 500, dtype=np.float32)
    out = aug.apply_rir(signal, np.array([1.0], dtype=np.float32))
    assert out.shape == signal.shape
    assert np.allclose(out, signal, atol=1e-5)


def test_apply_rir_preserves_length_and_bounded() -> None:
    rng = np.random.default_rng(3)
    signal = 0.3 * rng.standard_normal(2000).astype(np.float32)
    rir = rng.standard_normal(200).astype(np.float32)
    out = aug.apply_rir(signal, rir)
    assert out.shape == signal.shape
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_build_augmentation_plan_deterministic_and_sized() -> None:
    a = aug.build_augmentation_plan(4, seed=42)
    b = aug.build_augmentation_plan(4, seed=42)
    assert len(a) == 4
    assert a == b
    assert all(5.0 <= s.snr_db <= 20.0 for s in a)
    assert [s.round_index for s in a] == [0, 1, 2, 3]


def test_build_augmentation_plan_zero_rounds() -> None:
    assert aug.build_augmentation_plan(0) == []


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
