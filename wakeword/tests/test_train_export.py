"""Tests for M4 orchestration pieces that don't need torch/tf:
model shape contract, augmentation script wiring, provenance collection, git commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import augment as augscript  # noqa: E402
import train as trainmod  # noqa: E402
from lib import export as export_lib  # noqa: E402
from lib import licenses, model as model_lib  # noqa: E402


# --- model contract --------------------------------------------------------------

def test_flatten_input_dim() -> None:
    assert model_lib.flatten_input_dim() == 16 * 96
    assert model_lib.flatten_input_dim(8, 40) == 320


def test_export_opset_and_lib_import() -> None:
    assert export_lib.OPSET >= 11  # sane onnx opset for the FC + sigmoid graph


# --- augmentation script wiring --------------------------------------------------

CONFIG = {"wakeword": {"id": "keube", "phrase": "keube"},
          "training": {"augmentation_rounds": 2, "seed": 1337}}


def test_augment_positives_writes_variants_and_manifest(tmp_path: Path) -> None:
    written_paths: list[Path] = []

    def read_fn(p: Path) -> np.ndarray:
        return np.linspace(-0.2, 0.2, 32000, dtype=np.float32)

    def write_fn(p: Path, sig: np.ndarray) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"wav")
        written_paths.append(p)

    positives = [Path("pos_000000_spk0001.wav"), Path("pos_000001_spk0002.wav")]
    noise = [np.random.default_rng(0).standard_normal(32000).astype(np.float32)]
    rir = [np.array([1.0, 0.3, 0.1], dtype=np.float32)]

    count, manifest = augscript.augment_positives(
        CONFIG, datasets_root=tmp_path, read_fn=read_fn, write_fn=write_fn,
        positive_files=positives, noise_pool=noise, rir_pool=rir, rounds=2,
    )
    assert count == 4  # 2 clips x 2 rounds
    assert len(written_paths) == 4
    assert manifest.is_file()
    result = licenses.run_gate(tmp_path, require_commercial=True)
    assert result.ok


def test_augment_positives_requires_source_clips(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        augscript.augment_positives(
            CONFIG, datasets_root=tmp_path,
            read_fn=lambda p: np.zeros(10, np.float32), write_fn=lambda p, s: None,
            positive_files=[],
        )


# --- provenance collection -------------------------------------------------------

def test_collect_training_datasets_from_lock_and_manifests(tmp_path: Path) -> None:
    licenses.update_lock(tmp_path, key="musan", version="v1.0",
                         archives={"http://x/musan.tar.gz": "sha-abc"})
    licenses.write_dataset_manifest(
        datasets_root=tmp_path, key="positives-keube", category="speech",
        dataset="synthetic-positives/en_US-libritts_r-medium",
        source_url="https://github.com/rhasspy/piper", license_id="CC-BY-4.0",
        commercial_use=True, dataset_version="en_US-libritts_r-medium",
    )
    entries = trainmod.collect_training_datasets(tmp_path)
    names = {e["name"] for e in entries}
    assert "musan" in names
    musan = next(e for e in entries if e["name"] == "musan")
    assert musan["version"] == "v1.0"
    assert musan["sha256"] == "sha-abc"
    assert any("libritts" in e.get("version", "") or "synthetic" in e["name"] for e in entries)


def test_git_commit_returns_string() -> None:
    assert isinstance(trainmod.git_commit(), str)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
