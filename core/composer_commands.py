"""Immediate composer @-commands (app actions, not LLM routing attachments)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.app_settings import reset_help_guidance_settings
from core.app_notification_types import AppNotificationRequest
from core.app_restart import restart_action_label, restart_prompt_body
from core.composer_command_defs import COMPOSER_COMMANDS, ComposerCommand


@dataclass(frozen=True)
class ComposerCommandResult:
    ok: bool
    dialog_title: str = ""
    dialog_message: str = ""
    notification: AppNotificationRequest | None = None


def execute_composer_command(command_id: str, *, window: Any | None = None) -> ComposerCommandResult:
    """Run a palette command."""
    cmd = command_id.strip().lower()
    if cmd == "reset_help_guidance":
        reset_help_guidance_settings()
        _sync_help_guidance_ui(window)
        return ComposerCommandResult(
            ok=True,
            notification=AppNotificationRequest(
                title="Restart to apply",
                body=restart_prompt_body(purpose="run the setup tour immediately"),
                action_label=restart_action_label(),
                action_id="restart_app",
                severity="warning",
                category="update",
            ),
        )
    return ComposerCommandResult(
        ok=False,
        dialog_title="Command failed",
        dialog_message=f"Unknown command: {command_id}",
    )


def _sync_help_guidance_ui(window: Any | None) -> None:
    if window is None:
        return
    settings = getattr(window, "_settings_view", None)
    cb = getattr(settings, "model_manager_hardware_suggestions_cb", None)
    if cb is not None:
        cb.blockSignals(True)
        cb.setChecked(False)
        cb.blockSignals(False)
    mm = getattr(window, "_model_manager_view", None)
    if mm is not None and hasattr(mm, "refresh_hardware_suggestions"):
        mm.refresh_hardware_suggestions()


__all__ = ["COMPOSER_COMMANDS", "ComposerCommand", "ComposerCommandResult", "execute_composer_command"]
