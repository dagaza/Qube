"""STT and TTS model resolution and guards."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from core import stt_models as sm
from core import tts_models as tm


def _paths_equal(left: Path | None, right: Path) -> None:
    """Compare paths on Windows where resolve() may use 8.3 short names."""
    assert left is not None
    assert os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def _run_in_tmp(fn):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prev = os.getcwd()
        os.chdir(tmp)
        try:
            with patch(
                "core.stt_models.models_root",
                return_value=tmp_path / "models",
            ), patch(
                "core.tts_models.models_root",
                return_value=tmp_path / "models",
            ), patch(
                "core.tts_models.install_root",
                return_value=tmp_path / "install",
            ):
                fn(tmp_path)
        finally:
            os.chdir(prev)


def test_stt_bundled_default_is_small():
    assert sm.resolve_active_stt_model_spec() == sm.BUNDLED_STT_MODEL_ID


def test_stt_custom_folder_allowed():
    def body(root: Path) -> None:
        stt_dir = Path(sm.get_stt_models_dir())
        custom = stt_dir / "my-whisper"
        custom.mkdir(parents=True)
        (custom / "model.bin").write_bytes(b"x")
        ok, _ = sm.validate_stt_model_path(str(custom.resolve()))
        assert ok
        with patch(
            "core.stt_models.get_stt_model_path",
            return_value=str(custom.resolve()),
        ), patch(
            "core.model_paths_pro_features.custom_stt_override_allowed",
            return_value=True,
        ):
            assert sm.resolve_active_stt_model_spec() == str(custom.resolve())

    _run_in_tmp(body)


def test_stt_skips_hf_cache_dirs_in_list():
    def body(root: Path) -> None:
        stt_dir = Path(sm.get_stt_models_dir())
        cache = stt_dir / "models--Systran--faster-whisper-small"
        cache.mkdir(parents=True)
        (cache / "model.bin").write_bytes(b"x")
        custom = stt_dir / "user-model"
        custom.mkdir()
        (custom / "model.bin").write_bytes(b"y")
        names = [e.display_name for e in sm.list_selectable_stt_models()]
        assert "user-model" in names
        assert not any(n.startswith("models--") for n in names)

    _run_in_tmp(body)


def test_bundled_whisper_present_finds_hf_snapshot_layout():
    def body(root: Path) -> None:
        stt_dir = Path(sm.get_stt_models_dir())
        snap = (
            stt_dir
            / "models--Systran--faster-whisper-small"
            / "snapshots"
            / "abc123"
        )
        snap.mkdir(parents=True)
        (snap / "model.bin").write_bytes(b"x")
        assert sm.bundled_whisper_present() is True
        assert sm.stt_model_available() is True
        _paths_equal(sm.resolve_bundled_whisper_load_path(), snap)

    _run_in_tmp(body)


def test_bundled_whisper_prefers_flat_small_over_hf_cache():
    def body(root: Path) -> None:
        stt_dir = Path(sm.get_stt_models_dir())
        snap = (
            stt_dir
            / "models--Systran--faster-whisper-small"
            / "snapshots"
            / "abc123"
        )
        snap.mkdir(parents=True)
        (snap / "model.bin").write_bytes(b"cache")
        flat = sm.bundled_whisper_dir()
        flat.mkdir(parents=True)
        (flat / "model.bin").write_bytes(b"flat")
        _paths_equal(sm.resolve_bundled_whisper_load_path(), flat)

    _run_in_tmp(body)


def test_bundled_whisper_present_finds_flat_small_layout():
    def body(root: Path) -> None:
        whisper_dir = sm.bundled_whisper_dir()
        whisper_dir.mkdir(parents=True)
        (whisper_dir / "model.bin").write_bytes(b"x")
        assert sm.bundled_whisper_present() is True
        assert whisper_dir == Path(sm.get_stt_models_dir()) / "small"

    _run_in_tmp(body)


def test_bundled_whisper_absent_without_weights():
    def body(_root: Path) -> None:
        assert sm.bundled_whisper_present() is False
        assert sm.stt_model_available() is False

    _run_in_tmp(body)


def test_tts_bundled_default_path():
    def body(root: Path) -> None:
        tts_dir = root / "models" / tm.TTS_SUBDIR
        tts_dir.mkdir(parents=True)
        bundled = tts_dir / tm.BUNDLED_DEFAULT_FILENAME
        bundled.write_bytes(b"x")
        (tts_dir / tm.BUNDLED_VOICES_FILENAME).write_bytes(b"v")
        assert tm.resolve_active_tts_path() == str(bundled.resolve())

    _run_in_tmp(body)


def test_tts_migrate_legacy_layout():
    def body(root: Path) -> None:
        legacy = root / "install" / "models" / tm.TTS_SUBDIR
        legacy.mkdir(parents=True)
        (legacy / tm.BUNDLED_DEFAULT_FILENAME).write_bytes(b"onnx")
        (legacy / tm.BUNDLED_VOICES_FILENAME).write_bytes(b"voices")
        assert tm.migrate_legacy_tts_layout() is True
        assert Path(tm.bundled_default_path()).is_file()
        assert Path(tm.bundled_voices_path()).is_file()

    _run_in_tmp(body)


def test_tts_custom_onnx_listed():
    def body(root: Path) -> None:
        tts_dir = Path(tm.get_tts_models_dir())
        bundled = tts_dir / tm.BUNDLED_DEFAULT_FILENAME
        bundled.write_bytes(b"a")
        (tts_dir / tm.BUNDLED_VOICES_FILENAME).write_bytes(b"b")
        custom = tts_dir / "en_US-lessac-medium.onnx"
        custom.write_bytes(b"c")
        (custom.with_suffix(".onnx.json")).write_text("{}")
        names = [e.display_name for e in tm.list_selectable_tts_models()]
        assert tm.BUNDLED_TTS_LABEL in names
        assert "en_US-lessac-medium.onnx" in names

    _run_in_tmp(body)


def test_tts_classify_architecture():
    def body(_root: Path) -> None:
        tts_dir = Path(tm.get_tts_models_dir())
        kokoro = tts_dir / tm.BUNDLED_DEFAULT_FILENAME
        kokoro.write_bytes(b"x")
        assert tm.classify_tts_architecture(str(kokoro.resolve())) == "kokoro"
        piper = tts_dir / "en_US-lessac-medium.onnx"
        piper.write_bytes(b"x")
        (piper.with_suffix(".onnx.json")).write_text("{}")
        assert tm.classify_tts_architecture(str(piper.resolve())) == "piper"
        unknown = tts_dir / "other-tts.onnx"
        unknown.write_bytes(b"x")
        assert tm.classify_tts_architecture(str(unknown.resolve())) is None

    _run_in_tmp(body)


def test_tts_validate_rejects_unsupported_onnx():
    def body(root: Path) -> None:
        tts_dir = Path(tm.get_tts_models_dir())
        unknown = tts_dir / "other-tts.onnx"
        unknown.write_bytes(b"x")
        ok, msg = tm.validate_tts_model_path(str(unknown.resolve()))
        assert not ok
        assert "Kokoro and Piper" in msg

    _run_in_tmp(body)


def test_tts_validate_piper_requires_json_sidecar():
    def body(root: Path) -> None:
        tts_dir = Path(tm.get_tts_models_dir())
        piper = tts_dir / "en_GB-alba-medium.onnx"
        piper.write_bytes(b"x")
        ok, msg = tm.validate_tts_model_path(str(piper.resolve()))
        assert not ok
        assert "Piper" in msg

    _run_in_tmp(body)


def test_tts_any_supported_on_disk():
    def body(root: Path) -> None:
        tts_dir = Path(tm.get_tts_models_dir())
        piper = tts_dir / "en_US-lessac-medium.onnx"
        piper.write_bytes(b"x")
        (piper.with_suffix(".onnx.json")).write_text("{}")
        assert tm.any_supported_tts_model_on_disk() is True

    _run_in_tmp(body)


def test_tts_resolve_boot_falls_back_to_piper():
    def body(root: Path) -> None:
        tts_dir = Path(tm.get_tts_models_dir())
        piper = tts_dir / "en_US-lessac-medium.onnx"
        piper.write_bytes(b"x")
        (piper.with_suffix(".onnx.json")).write_text("{}")
        boot = tm.resolve_boot_tts_path()
        assert boot == str(piper.resolve())

    _run_in_tmp(body)
