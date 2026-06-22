"""Tests for bootstrap search preset helpers."""

from __future__ import annotations

from unittest.mock import patch

from core.bootstrap_manifest import BootstrapModelId
from core.bootstrap_search_models import (
    balanced_search_preset_present,
    embedding_preset_cached_on_disk,
)
from core.embedding_modes import DEFAULT_MODE


def test_balanced_search_preset_present_uses_cache_marker():
    from core.embedding_models import clear_embedding_availability_cache, mark_embedding_preset_available

    clear_embedding_availability_cache()
    with patch(
        "core.bootstrap_search_models.embedding_preset_cached_on_disk",
        return_value=False,
    ), patch(
        "core.embedding_models.gguf_override_available",
        return_value=False,
    ):
        assert balanced_search_preset_present() is False
        mark_embedding_preset_available(DEFAULT_MODE)
        assert balanced_search_preset_present() is True


def test_model_is_present_for_balanced_search_preset():
    from core.bootstrap_download import model_is_present

    with patch(
        "core.bootstrap_search_models.balanced_search_preset_present",
        return_value=True,
    ):
        assert model_is_present(BootstrapModelId.SEARCH_PRESET_BALANCED) is True


def test_embedding_preset_cached_on_disk_checks_hf_hub_layout():
    with patch("pathlib.Path.is_dir", return_value=True), patch(
        "pathlib.Path.rglob",
        return_value=[],
    ):
        assert embedding_preset_cached_on_disk("balanced") is False
