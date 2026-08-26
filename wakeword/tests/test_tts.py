"""Tests for the Piper synthesis PLAN + orchestration (lib/tts.py).

The plan (speaker spread, per-clip variation, filenames) is deterministic and tested
here; actual Piper synthesis is injected via a fake ``synth_fn`` so no voice is needed.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib import tts  # noqa: E402


def test_spread_speaker_ids_endpoints_and_spread() -> None:
    ids = tts.spread_speaker_ids(904, 5)
    assert len(ids) == 5
    assert ids[0] == 0
    assert ids[-1] == 903
    assert ids == sorted(ids)


def test_spread_speaker_ids_single_speaker() -> None:
    assert tts.spread_speaker_ids(1, 4) == [0, 0, 0, 0]


def test_spread_speaker_ids_cycles_when_over_subscribed() -> None:
    ids = tts.spread_speaker_ids(3, 6)
    assert len(ids) == 6
    assert all(0 <= i < 3 for i in ids)


def test_build_synthesis_plan_length_and_determinism() -> None:
    plan_a = tts.build_synthesis_plan(20, num_speakers=100)
    plan_b = tts.build_synthesis_plan(20, num_speakers=100)
    assert len(plan_a) == 20
    assert plan_a == plan_b  # deterministic


def test_build_synthesis_plan_varies_across_axes() -> None:
    plan = tts.build_synthesis_plan(30, num_speakers=100)
    assert len({p.speaker_id for p in plan}) > 1
    assert len({p.length_scale for p in plan}) > 1
    assert len({p.noise_scale for p in plan}) > 1
    assert len({p.noise_w for p in plan}) > 1


def test_clip_filename_is_sortable_and_descriptive() -> None:
    params = tts.SynthesisParams(speaker_id=7, length_scale=1.0, noise_scale=0.6, noise_w=0.8)
    name = tts.clip_filename("pos", 42, params)
    assert name == "pos_000042_spk0007.wav"


def _fake_synth(phrase: str, params: tts.SynthesisParams, out_path: Path) -> None:
    out_path.write_bytes(b"fake-wav")


def test_synthesize_clips_writes_all_and_returns_paths(tmp_path: Path) -> None:
    plan = tts.build_synthesis_plan(5, num_speakers=50)
    written = tts.synthesize_clips("keube", tmp_path, plan, synth_fn=_fake_synth, prefix="pos")
    assert len(written) == 5
    assert all(p.exists() for p in written)
    assert len(set(written)) == 5  # unique filenames


def test_write_silent_wav_is_valid_16k_mono(tmp_path: Path) -> None:
    out = tmp_path / "silence.wav"
    tts.write_silent_wav(out, seconds=0.5)
    with wave.open(str(out), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == tts.TARGET_SR
        assert wf.getnframes() == int(0.5 * tts.TARGET_SR)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
