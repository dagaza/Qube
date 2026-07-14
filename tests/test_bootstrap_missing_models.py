"""Tests for bootstrap missing-model guards (#47–#49)."""

from __future__ import annotations

from unittest.mock import patch

from core.bootstrap_manifest import BootstrapModelId
from core.bootstrap_missing_models import (
    ACTION_OPEN_SETTINGS_AI_COGNITION,
    ACTION_OPEN_SETTINGS_KNOWLEDGE_EMBEDDING,
    ACTION_OPEN_SETTINGS_VOICE_STT,
    ACTION_OPEN_SETTINGS_VOICE_TTS,
    guard_enable_embedding_feature,
    guard_enable_memory_enrichment,
    guard_enable_stt,
    guard_enable_tts,
    guard_library_upload,
    missing_embedding_notification,
    missing_stt_notification,
)


def test_guard_enable_stt_blocks_when_missing():
    with patch("core.bootstrap_missing_models.stt_model_available", return_value=False):
        allowed, event = guard_enable_stt(True)
    assert allowed is False
    assert event is not None
    assert event.action_id == ACTION_OPEN_SETTINGS_VOICE_STT


def test_guard_enable_stt_allows_disable_without_model():
    with patch("core.bootstrap_missing_models.stt_model_available", return_value=False):
        allowed, event = guard_enable_stt(False)
    assert allowed is True
    assert event is None


def test_guard_enable_tts_notification_action():
    with patch("core.bootstrap_missing_models.tts_model_available", return_value=False):
        allowed, event = guard_enable_tts(True)
    assert allowed is False
    assert event is not None
    assert event.action_id == ACTION_OPEN_SETTINGS_VOICE_TTS


def test_guard_embedding_feature_and_library_upload():
    with patch("core.bootstrap_missing_models.embedding_model_available", return_value=False):
        allowed, event = guard_enable_embedding_feature(True)
        assert allowed is False
        assert event is not None
        assert event.action_id == ACTION_OPEN_SETTINGS_KNOWLEDGE_EMBEDDING

        upload_ok, upload_event = guard_library_upload()
        assert upload_ok is False
        assert upload_event is not None
        assert upload_event.action_id == event.action_id


def test_guard_memory_enrichment_requires_cognition():
    with patch("core.bootstrap_missing_models.cognition_model_present", return_value=False):
        allowed, event = guard_enable_memory_enrichment(True)
    assert allowed is False
    assert event is not None
    assert event.action_id == ACTION_OPEN_SETTINGS_AI_COGNITION


def test_missing_embedding_notification_mentions_search_quality():
    event = missing_embedding_notification()
    assert "Search quality" in event.body
    assert "Connect to the internet" in event.body
    assert "Nomic" not in event.body


def test_guard_passes_when_models_present():
    with (
        patch("core.bootstrap_missing_models.stt_model_available", return_value=True),
        patch("core.bootstrap_missing_models.tts_model_available", return_value=True),
        patch("core.bootstrap_missing_models.embedding_model_available", return_value=True),
        patch("core.bootstrap_missing_models.cognition_model_present", return_value=True),
    ):
        assert guard_enable_stt(True) == (True, None)
        assert guard_enable_tts(True) == (True, None)
        assert guard_enable_embedding_feature(True) == (True, None)
        assert guard_enable_memory_enrichment(True) == (True, None)
