"""Embedding model resolution and guards."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from core import embedding_models as em


def _run_in_tmp(fn):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prev = os.getcwd()
        os.chdir(tmp)
        try:
            with patch(
                "core.embedding_models.models_root",
                return_value=tmp_path / "models",
            ), patch(
                "core.embedding_models.install_root",
                return_value=tmp_path / "install",
            ):
                fn(tmp_path)
        finally:
            os.chdir(prev)


def _bundled_path(root: Path) -> Path:
    return root / "models" / em.EMBEDDING_SUBDIR / em.BUNDLED_DEFAULT_FILENAME


def test_bundled_default_path_uses_embedding_subdir():
    def body(root: Path) -> None:
        emb_dir = root / "models" / em.EMBEDDING_SUBDIR
        emb_dir.mkdir(parents=True, exist_ok=True)
        bundled = emb_dir / em.BUNDLED_DEFAULT_FILENAME
        bundled.write_bytes(b"x")
        with patch(
            "core.embedding_models.get_embedding_models_dir",
            return_value=str(emb_dir),
        ):
            assert em.bundled_default_path() == str(bundled.resolve())
            assert em.embedding_model_available() is True

    _run_in_tmp(body)


def test_bundled_default_is_protected():
    def body(root: Path) -> None:
        bundled = _bundled_path(root)
        bundled.parent.mkdir(parents=True, exist_ok=True)
        bundled.write_bytes(b"x")
        assert em.is_protected_embedding_model(str(bundled))
        custom = bundled.parent / "custom-embed.gguf"
        custom.write_bytes(b"y")
        assert not em.is_protected_embedding_model(str(custom))

    _run_in_tmp(body)


def test_resolve_falls_back_to_bundled_when_override_invalid():
    def body(root: Path) -> None:
        bundled = _bundled_path(root)
        bundled.parent.mkdir(parents=True, exist_ok=True)
        bundled.write_bytes(b"x")
        with patch(
            "core.embedding_models.get_embedding_model_path",
            return_value=str(root / "missing.gguf"),
        ):
            assert Path(em.resolve_active_embedding_path()).resolve() == bundled.resolve()

    _run_in_tmp(body)


def test_embedding_dir_model_allowed():
    def body(root: Path) -> None:
        root.joinpath("models").mkdir()
        emb_dir = em.get_embedding_models_dir()
        custom = Path(emb_dir) / "custom-embed.gguf"
        custom.write_bytes(b"z" * 100)
        ok, _ = em.validate_embedding_model_path(str(custom))
        assert ok
        with patch(
            "core.embedding_models.get_embedding_model_path",
            return_value=str(custom.resolve()),
        ):
            assert em.resolve_active_embedding_path() == str(custom.resolve())

    _run_in_tmp(body)


def test_list_includes_bundled_first():
    def body(root: Path) -> None:
        bundled = _bundled_path(root)
        bundled.parent.mkdir(parents=True, exist_ok=True)
        bundled.write_bytes(b"a")
        entries = em.list_selectable_embedding_models()
        assert entries[0].is_bundled_default
        assert not entries[0].is_deletable

    _run_in_tmp(body)


def test_migrate_legacy_embedding_layout_copies_from_install_models():
    def body(root: Path) -> None:
        legacy = root / "install" / "models" / em.BUNDLED_DEFAULT_FILENAME
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(b"legacy-bytes")
        assert em.migrate_legacy_embedding_layout() is True
        target = Path(em.bundled_default_path())
        assert target.is_file()
        assert target.read_bytes() == b"legacy-bytes"

    _run_in_tmp(body)


def test_migrate_stale_embedding_override_clears_invalid_path():
    legacy = "/tmp/models/embedding/old.gguf"

    with patch(
        "core.embedding_models.get_embedding_model_path",
        return_value=legacy,
    ), patch(
        "core.embedding_models.validate_embedding_model_path",
        return_value=(False, "invalid"),
    ), patch("core.app_settings.set_embedding_model_path") as set_path:
        assert em.migrate_stale_embedding_override() is True
        set_path.assert_called_once_with("")
