"""Tests for training spec + data sampling (lib/training.py) — pure NumPy, no torch."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib import training  # noqa: E402

CONFIG = {"training": {"examples": 50000, "steps": 50000, "false_penalty": 2500,
                       "layer_dim": 32, "seed": 1337}}


def test_build_training_spec_full() -> None:
    spec = training.build_training_spec(CONFIG)
    assert (spec.examples, spec.steps, spec.false_penalty) == (50000, 50000, 2500)


def test_build_training_spec_pilot_caps() -> None:
    spec = training.build_training_spec(CONFIG, pilot=True)
    assert spec.examples == 5000
    assert spec.steps == 10000


def test_build_training_spec_pilot_overrides() -> None:
    spec = training.build_training_spec(CONFIG, pilot=True, pilot_overrides={"examples": 500, "steps": 800})
    assert spec.examples == 500 and spec.steps == 800


def test_negative_loss_weight_scales_and_caps() -> None:
    assert training.negative_loss_weight(0) == 1.0
    assert training.negative_loss_weight(2500) == pytest.approx(2.0)
    assert training.negative_loss_weight(10_000_000) == 20.0  # capped


def test_sample_batch_shapes_and_balance() -> None:
    rng = np.random.default_rng(0)
    pos = np.ones((10, 16, 96), dtype=np.float32)
    neg = np.zeros((10, 16, 96), dtype=np.float32)
    x, y = training.sample_batch(pos, neg, batch_size=64, rng=rng)
    assert x.shape == (64, 16, 96)
    assert y.shape == (64, 1)
    assert int(y.sum()) == 32  # class-balanced


def test_sample_batch_requires_both_classes() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        training.sample_batch(np.ones((0, 16, 96), np.float32), np.ones((5, 16, 96), np.float32), 8, rng)


def test_sample_weights_positive_vs_negative() -> None:
    labels = np.array([[1.0], [0.0], [1.0], [0.0]], dtype=np.float32)
    w = training.sample_weights(labels, false_penalty=2500)
    assert w.shape == (4, 1)
    assert w[0, 0] == 1.0  # positive
    assert w[1, 0] == pytest.approx(2.0)  # negative weighted


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
