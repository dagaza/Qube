"""Tests for shell bootstrap and feature download prompts."""

from __future__ import annotations

from unittest.mock import patch

from core.bootstrap_feasibility import can_proceed_with_selection, build_session_assessment
from core.bootstrap_hf_metadata import BootstrapSizeSource, ResolvedBootstrapSize
from core.bootstrap_manifest import BootstrapModelId
from core.bootstrap_selection import (
    effective_bootstrap_selection,
    get_selected_model_ids,
    is_bootstrap_completed,
    save_bootstrap_selection,
)
from ui.bootstrap_feature_prompts import (
    format_bootstrap_model_confirm_body,
    main_llm_model_available,
)


def _minimal_assessment():
    resolved = {
        model_id: ResolvedBootstrapSize(
            model_id=model_id,
            size_bytes=1000,
            source=BootstrapSizeSource.ESTIMATE,
            detail="test",
        )
        for model_id in BootstrapModelId
    }
    return build_session_assessment(resolved=resolved)


def test_can_proceed_with_empty_selection_when_allowed():
    assessment = _minimal_assessment()
    ok, msg = can_proceed_with_selection(set(), assessment, allow_empty=True)
    assert ok is True
    assert msg == ""


def test_can_proceed_with_empty_selection_blocked_by_default():
    assessment = _minimal_assessment()
    ok, msg = can_proceed_with_selection(set(), assessment)
    assert ok is False
    assert "at least one model" in msg.lower()


def test_effective_bootstrap_selection_empty_after_shell_install(monkeypatch):
    import tempfile
    from pathlib import Path

    from core.settings_store import SettingsStore

    schema_path = (
        Path(__file__).resolve().parent.parent / "assets" / "config" / "settings.schema.json"
    )
    with tempfile.TemporaryDirectory() as tmp:
        store = SettingsStore(user_path=Path(tmp) / "settings.json", schema_path=schema_path)
        monkeypatch.setattr("core.bootstrap_selection.get_settings_store", lambda: store)
        monkeypatch.setattr("core.bootstrap_selection.apply_bootstrap_selection", lambda _s: None)

        save_bootstrap_selection(set())

        assert is_bootstrap_completed()
        assert get_selected_model_ids() == set()
        assert effective_bootstrap_selection() == set()


def test_format_shell_install_warning_message():
    from core.bootstrap_feasibility import format_shell_install_warning_message

    body = format_shell_install_warning_message()
    assert "barebones" in body.lower()
    assert "Sidecar" in body
    assert "instability" in body.lower()
    assert "crashes" in body.lower()


def test_format_bootstrap_model_confirm_body():
    body = format_bootstrap_model_confirm_body(
        BootstrapModelId.WHISPER_SMALL,
        feature_label="Voice input",
    )
    assert "Voice input" in body
    assert "Whisper" in body
    assert "Download now" in body


def test_main_llm_model_available_external_mode(monkeypatch):
    monkeypatch.setattr("core.app_settings.get_engine_mode", lambda: "external")
    assert main_llm_model_available() is True
