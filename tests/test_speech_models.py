"""STT and TTS model resolution and guards."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from core import stt_models as sm
from core import tts_models as tm


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
        names = [e.display_name for e in tm.list_selectable_tts_models()]
        assert tm.BUNDLED_TTS_LABEL in names
        assert "en_US-lessac-medium.onnx" in names

    _run_in_tmp(body)
