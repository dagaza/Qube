"""Tests for embedding mode switch UX helpers."""

from __future__ import annotations

from unittest.mock import patch

from core.bootstrap_search_models import (
    embedding_mode_switch_needs_download,
    format_embedding_mode_switch_confirm_body,
    format_search_preset_download_failure,
    is_likely_embedding_load_failure,
    search_preset_size_bytes,
)


def test_search_preset_size_bytes_per_mode():
    assert search_preset_size_bytes("fast") < search_preset_size_bytes("balanced")
    assert search_preset_size_bytes("balanced") < search_preset_size_bytes("power")


def test_format_embedding_mode_switch_confirm_body_mentions_download_when_needed():
    with patch(
        "core.bootstrap_search_models.embedding_mode_switch_needs_download",
        return_value=True,
    ):
        body = format_embedding_mode_switch_confirm_body("power")
    assert "not on this device yet" in body
    assert "download when online" in body
    assert "Continue?" in body


def test_format_embedding_mode_switch_confirm_body_without_download_note():
    with patch(
        "core.bootstrap_search_models.embedding_mode_switch_needs_download",
        return_value=False,
    ):
        body = format_embedding_mode_switch_confirm_body("balanced")
    assert "not on this device yet" not in body
    assert "reprocess your library" in body.lower()


def test_embedding_mode_switch_needs_download_skips_when_gguf_override():
    with patch(
        "core.embedding_models.gguf_override_available",
        return_value=True,
    ):
        assert embedding_mode_switch_needs_download("power") is False


def test_format_search_preset_download_failure_mentions_search_quality():
    msg = format_search_preset_download_failure("balanced")
    assert "Balanced" in msg
    assert "Advanced embedding" in msg


def test_is_likely_embedding_load_failure():
    assert is_likely_embedding_load_failure("fastembed download failed") is True
    assert is_likely_embedding_load_failure("disk full") is False
