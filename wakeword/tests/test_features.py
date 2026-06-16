"""Tests for M2 feature-precompute helpers.

Covers the pure-numpy logic (clip stacking, shard merge) and the provenance manifest
writer. The openWakeWord embedding step itself needs the ONNX models + training env
and is exercised separately, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from lib import audio, features, licenses  # noqa: E402


def test_samples_to_clips_drops_remainder() -> None:
    signal = np.ones(32000 * 2 + 5, dtype=np.float32) * 0.5
    clips = audio.samples_to_clips(signal, clip_size=32000)
    assert clips.shape == (2, 32000)
    assert clips.dtype == np.int16
    # 0.5 * 32767 rounds toward zero on cast
    assert clips[0, 0] == np.int16(0.5 * 32767)


def test_samples_to_clips_too_short_is_empty() -> None:
    clips = audio.samples_to_clips(np.ones(100, dtype=np.float32), clip_size=32000)
    assert clips.shape == (0, 32000)


def test_iter_clip_batches_batches_and_carries(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each fake file yields 1.5 clips worth of samples; the carry buffer must join
    # them so no audio is lost across file boundaries.
    clip_size = 100
    fake = {f"f{i}": np.ones(int(clip_size * 1.5), dtype=np.float32) for i in range(4)}
    monkeypatch.setattr(audio, "read_mono_16k", lambda p: fake[str(p)])

    batches = list(
        audio.iter_clip_batches(list(fake.keys()), clip_size=clip_size, batch_clips=2)
    )
    total_rows = sum(b.shape[0] for b in batches)
    # 4 files * 1.5 clips = 6 clips total
    assert total_rows == 6
    assert all(b.shape[1] == clip_size for b in batches)
    # all full batches except possibly the last
    assert all(b.shape[0] == 2 for b in batches[:-1])


def test_merge_shards_concatenates_in_order(tmp_path: Path) -> None:
    shard_paths = []
    for i in range(3):
        arr = np.full((2, 16, 96), float(i), dtype=np.float32)
        p = tmp_path / f"shard_{i}.npy"
        np.save(p, arr)
        shard_paths.append(p)

    out = tmp_path / "merged.npy"
    rows = features.merge_shards(shard_paths, out)
    assert rows == 6

    merged = np.load(out)
    assert merged.shape == (6, 16, 96)
    assert merged[0, 0, 0] == 0.0
    assert merged[2, 0, 0] == 1.0
    assert merged[4, 0, 0] == 2.0


def test_merge_shards_rejects_wrong_shape(tmp_path: Path) -> None:
    p = tmp_path / "bad.npy"
    np.save(p, np.zeros((2, 8, 96), dtype=np.float32))
    with pytest.raises(ValueError):
        features.merge_shards([p], tmp_path / "out.npy")


def test_compute_features_streams_via_fake_extractor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake audio + embedding so we exercise the shard/merge plumbing without ONNX.
    clip_size = 100
    files = [f"f{i}" for i in range(10)]
    monkeypatch.setattr(
        audio, "read_mono_16k", lambda p: np.ones(clip_size, dtype=np.float32)
    )

    class FakeExtractor:
        def embed(self, clips: np.ndarray) -> np.ndarray:
            return np.zeros((clips.shape[0], 16, 96), dtype=np.float32)

    out = tmp_path / "features" / "neg.npy"
    rows = features.compute_features(
        files,
        out,
        extractor=FakeExtractor(),
        clip_size=clip_size,
        batch_clips=2,
        rows_per_shard=3,
        shard_dir=tmp_path / "_shards",
    )
    assert rows == 10
    assert np.load(out).shape == (10, 16, 96)
    # shards cleaned up
    assert not (tmp_path / "_shards").exists()


def test_compute_features_respects_max_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_size = 100
    files = [f"f{i}" for i in range(10)]
    monkeypatch.setattr(
        audio, "read_mono_16k", lambda p: np.ones(clip_size, dtype=np.float32)
    )

    class FakeExtractor:
        def embed(self, clips: np.ndarray) -> np.ndarray:
            return np.zeros((clips.shape[0], 16, 96), dtype=np.float32)

    out = tmp_path / "neg.npy"
    rows = features.compute_features(
        files, out, extractor=FakeExtractor(), clip_size=clip_size,
        batch_clips=4, rows_per_shard=100, max_rows=5,
    )
    assert rows == 5
    assert np.load(out).shape == (5, 16, 96)


def test_write_manifest_passes_license_gate(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    feat = datasets_root / "features" / "neg.npy"
    feat.parent.mkdir(parents=True)
    np.save(feat, np.zeros((4, 16, 96), dtype=np.float32))

    licenses.write_manifest(
        feat,
        datasets_root=datasets_root,
        dataset="LibriSpeech",
        source_url="https://www.openslr.org/12",
        license_id="CC-BY-4.0",
        commercial_use=True,
        attribution="LibriSpeech CC-BY-4.0.",
    )

    result = licenses.run_gate(datasets_root, require_commercial=True)
    assert result.ok
    assert result.checked == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
