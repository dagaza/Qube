"""Tests for missing native model autoload handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from core.native_model_autoload import (
    ACTION_OPEN_MODELS,
    ACTION_OPEN_MODELS_REDOWNLOAD,
    clear_stale_native_model_path,
    decode_redownload_action_id,
    encode_redownload_action_id,
    evaluate_native_model_refresh,
    lookup_hf_redownload_target,
    missing_autoload_model_notification,
)


def test_encode_decode_redownload_action_id():
    action = encode_redownload_action_id("bartowski/Llama-GGUF", "model-q4_0.gguf")
    assert action.startswith(ACTION_OPEN_MODELS_REDOWNLOAD)
    decoded = decode_redownload_action_id(action)
    assert decoded == ("bartowski/Llama-GGUF", "model-q4_0.gguf")


def test_missing_autoload_notification_offers_redownload_when_provenance_known():
    event = missing_autoload_model_notification(
        display_name="model-q4_0.gguf",
        redownload_repo_id="org/repo",
        redownload_filename="model-q4_0.gguf",
    )
    assert event.action_label == "Re-download"
    assert decode_redownload_action_id(event.action_id or "") == (
        "org/repo",
        "model-q4_0.gguf",
    )


def test_missing_autoload_notification_falls_back_to_model_manager():
    event = missing_autoload_model_notification(display_name="gone.gguf")
    assert event.action_id == ACTION_OPEN_MODELS
    assert event.action_label == "Open Model Manager"


def test_evaluate_native_model_refresh_clears_missing_path(tmp_path: Path):
    missing = tmp_path / "removed.gguf"
    saved_path = str(missing)
    cleared: list[str] = []

    with patch(
        "core.native_model_autoload.lookup_hf_redownload_target",
        return_value=("org/model", "removed.gguf"),
    ), patch(
        "core.native_model_autoload.clear_stale_native_model_path",
        side_effect=lambda p: cleared.append(p) or True,
    ):
        outcome = evaluate_native_model_refresh(saved_path, autoload=True)

    assert outcome.notify_user is True
    assert outcome.missing_display_name == "removed.gguf"
    assert outcome.redownload_repo_id == "org/model"
    assert outcome.cleared_stale_path is True
    assert cleared == [saved_path]


def test_lookup_hf_provenance_by_basename(tmp_path: Path):
    old_path = str(tmp_path / "old/path/model.gguf")
    new_path = str(tmp_path / "new/path/model.gguf")

    class _Store:
        def get_model_hf_provenance(self, local_path: str) -> str | None:
            return None

        def load_model_hf_provenance_map(self) -> dict[str, str]:
            return {old_path: "org/catalog"}

    with patch("core.native_model_autoload.SystemCapabilitiesStore", _Store):
        found = lookup_hf_redownload_target(new_path)
    assert found == ("org/catalog", "model.gguf")


def test_clear_stale_native_model_path_clears_settings(tmp_path: Path, monkeypatch):
    path = tmp_path / "model.gguf"
    settings: dict[str, object] = {"qube.native.modelPath": str(path)}

    class _Store:
        def get(self, key, default=None):
            return settings.get(key, default)

        def set(self, key, value, force=False):
            settings[key] = value

    monkeypatch.setattr("core.app_settings._store", lambda: _Store())
    monkeypatch.setattr(
        "core.native_model_autoload.SystemCapabilitiesStore.remove_model_hf_provenance",
        lambda self, p: None,
    )

    assert clear_stale_native_model_path(str(path)) is True
    assert settings["qube.native.modelPath"] == ""
