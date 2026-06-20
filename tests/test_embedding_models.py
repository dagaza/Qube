"""Embedding model path resolution tests (GGUF advanced override)."""
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
            ):
                fn(tmp_path)
        finally:
            os.chdir(prev)


def test_embedding_dir_model_allowed():
    def body(root: Path) -> None:
        emb_dir = em.get_embedding_models_dir()
        custom = Path(emb_dir) / "custom-embed.gguf"
        custom.write_bytes(b"z" * 100)
        ok, _ = em.validate_embedding_model_path(str(custom))
        assert ok
        with patch(
            "core.embedding_models.get_embedding_model_path",
            return_value=str(custom.resolve()),
        ):
            assert em.resolve_active_gguf_path() == str(custom.resolve())

    _run_in_tmp(body)


def test_list_selectable_scans_embedding_dir():
    def body(root: Path) -> None:
        emb_dir = Path(em.get_embedding_models_dir())
        (emb_dir / "one.gguf").write_bytes(b"a")
        entries = em.list_selectable_embedding_models()
        assert len(entries) == 1
        assert entries[0].is_deletable

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
