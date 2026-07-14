"""Tests for the model_card builder (lib/model_card.py)."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib import model_card  # noqa: E402

CONFIG = {
    "wakeword": {"id": "hey_keube", "display_name": "Hey Qube", "phrase": "hey_keube"},
    "training": {"examples": 50000, "steps": 50000, "false_penalty": 2500,
                 "layer_dim": 32, "augmentation_rounds": 2, "seed": 1337},
    "data": {"sample_rate": 16000},
    "provenance": {"tier": "commercial", "oww_commit": "abc123"},
}


def test_config_hash_is_order_independent() -> None:
    a = model_card.canonical_config_hash({"x": 1, "y": 2})
    b = model_card.canonical_config_hash({"y": 2, "x": 1})
    assert a == b
    assert a != model_card.canonical_config_hash({"x": 1, "y": 3})


def test_build_model_card_core_fields() -> None:
    card = model_card.build_model_card(
        CONFIG, version="v0.1", git_commit="deadbeef",
        training_datasets=[{"name": "musan", "version": "v1.0", "license": "CC-BY-4.0"}],
        metrics={"validation_false_positive_rate": 0.01}, hardware="RTX 4090",
        duration_seconds=123.4,
    )
    assert card["id"] == "hey_keube"
    assert card["runtime"]["input_shape"] == [16, 96]
    assert card["runtime"]["output_shape"] == [1]
    assert card["training"]["false_penalty"] == 2500
    assert card["provenance"]["git_commit"] == "deadbeef"
    assert card["provenance"]["config_hash"]
    assert card["training_datasets"][0]["name"] == "musan"
    assert card["metrics"]["validation_false_positive_rate"] == 0.01


def test_commercial_flag_true_only_when_passed() -> None:
    card = model_card.build_model_card(
        CONFIG, version="v0.1", git_commit="x", training_datasets=[], license_audit="passed"
    )
    assert card["license"]["tier"] == "commercial"
    assert card["license"]["commercial_use"] is True


def test_commercial_flag_false_when_skipped() -> None:
    card = model_card.build_model_card(
        CONFIG, version="v0.1", git_commit="x", training_datasets=[], license_audit="skipped"
    )
    assert card["license"]["commercial_use"] is False


def test_personal_tier_is_never_commercial() -> None:
    cfg = {**CONFIG, "provenance": {"tier": "personal"}}
    card = model_card.build_model_card(
        cfg, version="v0.1", git_commit="x", training_datasets=[], license_audit="passed"
    )
    assert card["license"]["commercial_use"] is False


def test_write_model_card_roundtrips(tmp_path: Path) -> None:
    import json

    card = model_card.build_model_card(CONFIG, version="v0.1", git_commit="x", training_datasets=[])
    out = model_card.write_model_card(card, tmp_path / "sub" / "model_card.json")
    assert out.is_file()
    assert json.loads(out.read_text())["id"] == "hey_keube"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
