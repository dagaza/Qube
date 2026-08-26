"""Tests for hard_negative_mining.py (M3 confusable synthesis)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import hard_negative_mining as hnm  # noqa: E402
from lib import licenses, tts  # noqa: E402

CONFIG = {
    "wakeword": {
        "id": "keube",
        "phrase": "keube",
        "adversarial_phrases": ["cube", "queue", "tube"],
    },
    "training": {"examples": 5000},
}


def _fake_synth(phrase: str, params: tts.SynthesisParams, out_path: Path) -> None:
    out_path.write_bytes(b"fake-wav")


def test_allocate_splits_evenly_with_remainder() -> None:
    assert hnm.allocate(10, 3) == [4, 3, 3]
    assert hnm.allocate(9, 3) == [3, 3, 3]
    assert hnm.allocate(2, 5) == [1, 1, 0, 0, 0]
    assert sum(hnm.allocate(97, 7)) == 97


def test_hard_negatives_for_config_merges_family_and_config() -> None:
    phrases = hnm.hard_negatives_for_config(CONFIG)
    assert "cube" in phrases  # config
    assert "youtube" in phrases  # library
    assert "keube" not in phrases  # never the wake phrase


def test_mine_hard_negatives_writes_clips_and_manifest(tmp_path: Path) -> None:
    written, manifest = hnm.mine_hard_negatives(
        CONFIG, datasets_root=tmp_path, count=30, synth_fn=_fake_synth, num_speakers=20
    )
    # total clips equal the requested budget (allocation is exhaustive)
    assert len(written) == 30
    out_dir = tmp_path / "speech" / "hard-negative" / "keube"
    assert out_dir.is_dir()
    # clips are grouped by confusable-phrase prefix
    assert len(list(out_dir.glob("neg_cube_*.wav"))) >= 1
    assert manifest.is_file()


def test_mined_manifest_passes_commercial_gate(tmp_path: Path) -> None:
    hnm.mine_hard_negatives(
        CONFIG, datasets_root=tmp_path, count=12, synth_fn=_fake_synth, num_speakers=8
    )
    result = licenses.run_gate(tmp_path, require_commercial=True)
    assert result.ok


def test_mine_raises_without_any_confusables() -> None:
    empty = {"wakeword": {"id": "zzz", "phrase": "zzz", "adversarial_phrases": []}}
    with pytest.raises(ValueError, match="hard-negative"):
        hnm.mine_hard_negatives(
            empty, datasets_root=Path("."), count=1, synth_fn=_fake_synth
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
