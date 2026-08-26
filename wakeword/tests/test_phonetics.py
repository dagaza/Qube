"""Tests for hard-negative / confusable phrase generation (lib/phonetics.py)."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib import phonetics  # noqa: E402


def test_normalize_phrase_handles_underscores_and_punctuation() -> None:
    assert phonetics.normalize_phrase("Hey_Keube!") == "hey keube"
    assert phonetics.normalize_phrase("  a  CUBE  ") == "a cube"
    assert phonetics.normalize_phrase("rubik's cube") == "rubiks cube"


def test_detect_family_from_spellings() -> None:
    for spelling in ("keube", "cube", "kyoob", "kay_oob", "hey_keube", "hey cube"):
        assert phonetics.detect_family(spelling) == "cube"


def test_detect_family_unknown_is_none() -> None:
    assert phonetics.detect_family("banana") is None
    assert phonetics.detect_family("") is None


def test_build_hard_negatives_includes_library_and_config() -> None:
    negs = phonetics.build_hard_negatives(
        "keube", adversarial_phrases=["cube", "my custom word"]
    )
    # config phrases come first and are preserved
    assert negs[0] == "cube"
    assert "my custom word" in negs
    # library confusables are merged in
    for expected in ("tube", "cute", "queue", "youtube", "cubed"):
        assert expected in negs


def test_build_hard_negatives_excludes_the_wake_phrase() -> None:
    negs = phonetics.build_hard_negatives("cube", adversarial_phrases=["cube", "tube"])
    assert "cube" not in negs
    assert "tube" in negs


def test_build_hard_negatives_dedupes_normalized() -> None:
    negs = phonetics.build_hard_negatives(
        "keube", adversarial_phrases=["Tube", "tube!", " TUBE "]
    )
    assert negs.count("tube") == 1


def test_build_hard_negatives_without_library() -> None:
    negs = phonetics.build_hard_negatives(
        "keube", adversarial_phrases=["tube"], include_library=False
    )
    assert negs == ["tube"]


def test_build_hard_negatives_extra_appended() -> None:
    negs = phonetics.build_hard_negatives(
        "keube", adversarial_phrases=[], extra=["zoombie"], include_library=False
    )
    assert negs == ["zoombie"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
