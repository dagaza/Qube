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
        ), patch(
            "core.model_paths_pro_features.custom_embedding_override_allowed",
            return_value=True,
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


def test_embedding_model_available_with_gguf_override():
    def body(root: Path) -> None:
        emb_dir = em.get_embedding_models_dir()
        custom = Path(emb_dir) / "custom-embed.gguf"
        custom.write_bytes(b"z" * 100)
        with patch(
            "core.embedding_models.get_embedding_model_path",
            return_value=str(custom.resolve()),
        ), patch(
            "core.model_paths_pro_features.custom_embedding_override_allowed",
            return_value=True,
        ):
            assert em.gguf_override_available() is True
            assert em.embedding_model_available() is True
            assert em.preset_embedder_ready() is False

    _run_in_tmp(body)


def test_preset_embedder_ready_requires_on_disk_cache():
    with patch("core.embedding_models.gguf_override_available", return_value=False), patch(
        "core.app_settings.get_embedding_mode", return_value="balanced"
    ), patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=False,
    ):
        assert em.preset_embedder_ready() is False
    with patch("core.embedding_models.gguf_override_available", return_value=False), patch(
        "core.app_settings.get_embedding_mode",
        return_value="balanced",
    ), patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=True,
    ):
        assert em.preset_embedder_ready() is True


def test_embedding_model_available_requires_on_disk_or_probe():
    with patch("core.embedding_models.gguf_override_available", return_value=False), patch(
        "core.app_settings.get_embedding_mode", return_value="balanced"
    ), patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=False,
    ), patch(
        "core.embedding_models.probe_embedding_preset_available",
        return_value=False,
    ):
        assert em.embedding_model_available() is False
    with patch("core.embedding_models.gguf_override_available", return_value=False), patch(
        "core.app_settings.get_embedding_mode",
        return_value="balanced",
    ), patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=True,
    ):
        assert em.embedding_model_available() is True


def test_probe_embedding_preset_available_marks_cache_on_success():
    em.clear_embedding_availability_cache()
    fake_backend = type("B", (), {"vector_dim": 512, "unload": lambda self: None})()
    fake_model = type("M", (), {"vector_dim": 512, "_backend": fake_backend})()
    with patch("core.embedding_models.gguf_override_available", return_value=False), patch(
        "core.bootstrap_search_download.download_embedding_preset_no_progress"
    ), patch(
        "rag.embedder.EmbeddingModel", return_value=fake_model
    ), patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        side_effect=[False, True, True, True],
    ):
        assert em.probe_embedding_preset_available(force=True) is True
    with patch("core.embedding_models.gguf_override_available", return_value=False), patch(
        "core.app_settings.get_embedding_mode",
        return_value="balanced",
    ), patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=True,
    ):
        assert em.preset_embedder_ready() is True
