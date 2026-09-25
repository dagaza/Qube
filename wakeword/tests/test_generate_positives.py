"""Tests for generate_positives.py (M3 positive synthesis orchestration).

Piper is injected via a fake ``synth_fn`` so the test needs no voice model. We assert
clip layout, count, and that the emitted provenance manifest passes the commercial gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import generate_positives as gp  # noqa: E402
from lib import licenses, tts  # noqa: E402

CONFIG = {
    "wakeword": {"id": "keube", "display_name": "Qube", "phrase": "keube"},
    "training": {"examples": 5000},
}


def _fake_synth(phrase: str, params: tts.SynthesisParams, out_path: Path) -> None:
    out_path.write_bytes(b"fake-wav")


def test_resolve_count_full_vs_pilot_vs_override() -> None:
    assert gp.resolve_count(CONFIG, pilot=False, count=None) == 5000
    assert gp.resolve_count(CONFIG, pilot=True, count=None) == gp.PILOT_DEFAULT_COUNT
    assert gp.resolve_count(CONFIG, pilot=True, count=42) == 42


def test_generate_positives_writes_clips_and_manifest(tmp_path: Path) -> None:
    written, manifest = gp.generate_positives(
        CONFIG, datasets_root=tmp_path, count=8, synth_fn=_fake_synth, num_speakers=50
    )
    assert len(written) == 8
    out_dir = tmp_path / "speech" / "positive" / "keube"
    assert out_dir.is_dir()
    assert len(list(out_dir.glob("pos_*.wav"))) == 8
    assert manifest.is_file()


def test_generated_positive_manifest_passes_commercial_gate(tmp_path: Path) -> None:
    gp.generate_positives(
        CONFIG, datasets_root=tmp_path, count=3, synth_fn=_fake_synth, num_speakers=10
    )
    result = licenses.run_gate(tmp_path, require_commercial=True)
    assert result.ok
    assert result.checked == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
