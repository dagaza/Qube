"""Auxiliary cognition model resolution and guards."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from core import auxiliary_cognition as ac


def _run_in_tmp(fn):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prev = os.getcwd()
        os.chdir(tmp)
        try:
            with patch(
                "core.auxiliary_cognition.models_root",
                return_value=tmp_path / "models",
            ):
                fn(tmp_path)
        finally:
            os.chdir(prev)


def _bundled_path(root: Path) -> Path:
    return root / ac.BUNDLED_DEFAULT_REL_PATH


def test_bundled_default_path_uses_cognition_subdir():
    def body(root: Path) -> None:
        cog_dir = root / "models" / "cognition"
        cog_dir.mkdir(parents=True, exist_ok=True)
        bundled = cog_dir / Path(ac.BUNDLED_DEFAULT_REL_PATH).name
        bundled.write_bytes(b"x")
        with patch(
            "core.auxiliary_cognition.get_cognition_models_dir",
            return_value=str(cog_dir),
        ):
            assert Path(ac.bundled_default_path()).resolve() == bundled.resolve()
            assert ac.cognition_model_available() is True

    _run_in_tmp(body)


def test_bundled_default_is_protected():
    def body(root: Path) -> None:
        bundled = _bundled_path(root)
        bundled.parent.mkdir(parents=True, exist_ok=True)
        bundled.write_bytes(b"x")
        assert ac.is_protected_cognition_model(str(bundled))
        custom = bundled.parent / "phi.gguf"
        custom.write_bytes(b"y")
        assert not ac.is_protected_cognition_model(str(custom))

    _run_in_tmp(body)


def test_resolve_falls_back_to_bundled_when_override_invalid():
    def body(root: Path) -> None:
        bundled = _bundled_path(root)
        bundled.parent.mkdir(parents=True, exist_ok=True)
        bundled.write_bytes(b"x")
        with patch(
            "core.auxiliary_cognition.get_sidecar_model_path",
            return_value=str(root / "missing.gguf"),
        ):
            assert Path(ac.resolve_active_cognition_path()).resolve() == bundled.resolve()

    _run_in_tmp(body)


def test_cognition_dir_model_allowed():
    def body(root: Path) -> None:
        (root / "models").mkdir()
        cog_dir = ac.get_cognition_models_dir()
        custom = Path(cog_dir) / "gemma-2b.gguf"
        custom.write_bytes(b"z" * 100)
        ok, _ = ac.validate_cognition_model_path(str(custom))
        assert ok
        with patch(
            "core.auxiliary_cognition.get_sidecar_model_path",
            return_value=str(custom.resolve()),
        ):
            assert ac.resolve_active_cognition_path() == str(custom.resolve())

    _run_in_tmp(body)


def test_list_includes_bundled_first():
    def body(root: Path) -> None:
        bundled = _bundled_path(root)
        bundled.parent.mkdir(parents=True, exist_ok=True)
        bundled.write_bytes(b"a")
        entries = ac.list_selectable_cognition_models()
        assert entries[0].is_bundled_default
        assert not entries[0].is_deletable

    _run_in_tmp(body)


def test_cognition_n_ctx_for_qwen3():
    assert ac.cognition_n_ctx_for_path("Qwen3-1.7B-Q6_K.gguf") == 4096


def test_migrate_stale_sidecar_override_clears_invalid_path():
    legacy = "/tmp/models/qwen2-0_5b-instruct-q4_k_m.gguf"

    with patch(
        "core.auxiliary_cognition.get_sidecar_model_path",
        return_value=legacy,
    ), patch(
        "core.auxiliary_cognition.validate_cognition_model_path",
        return_value=(False, "invalid"),
    ), patch("core.app_settings.set_sidecar_model_path") as set_path:
        assert ac.migrate_stale_sidecar_override() is True
        set_path.assert_called_once_with("")


def test_migrate_stale_sidecar_override_noop_when_valid():
    with patch(
        "core.auxiliary_cognition.get_sidecar_model_path",
        return_value="/models/cognition/Qwen3-1.7B-Q6_K.gguf",
    ), patch(
        "core.auxiliary_cognition.validate_cognition_model_path",
        return_value=(True, ""),
    ), patch("core.app_settings.set_sidecar_model_path") as set_path:
        assert ac.migrate_stale_sidecar_override() is False
        set_path.assert_not_called()


def test_size_cap_allows_large_cognition_model():
    def body(root: Path) -> None:
        cog_dir = Path(ac.get_cognition_models_dir())
        custom = cog_dir / "large.gguf"
        custom.write_bytes(b"x")

        under_cap = ac.MAX_COGNITION_FILE_BYTES - 1
        with patch("os.path.getsize", return_value=under_cap):
            ok, _ = ac.validate_cognition_model_path(str(custom.resolve()))
            assert ok

        over_cap = ac.MAX_COGNITION_FILE_BYTES + 1
        with patch("os.path.getsize", return_value=over_cap):
            ok, msg = ac.validate_cognition_model_path(str(custom.resolve()))
            assert not ok
            assert "2048" in msg

    _run_in_tmp(body)
