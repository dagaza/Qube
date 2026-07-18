"""Immediate composer @-commands (app actions, not LLM routing attachments)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.app_settings import reset_help_guidance_settings
from core.app_notification_types import AppNotificationRequest
from core.app_restart import restart_action_label, restart_prompt_body


@dataclass(frozen=True)
class ComposerCommand:
    id: str
    label: str
    description: str
    requires_confirmation: bool = False
    confirmation_title: str = ""
    confirmation_message: str = ""


@dataclass(frozen=True)
class ComposerCommandResult:
    ok: bool
    dialog_title: str = ""
    dialog_message: str = ""
    notification: AppNotificationRequest | None = None


COMPOSER_COMMANDS: tuple[ComposerCommand, ...] = (
    ComposerCommand(
        id="reset_help_guidance",
        label="Reset Help & Guidance",
        description="Run setup tour on next launch; turn off Model Manager hints",
        requires_confirmation=True,
        confirmation_title="Reset Help & Guidance",
        confirmation_message=(
            "This will restore Help & Guidance to defaults:\n\n"
            "• The Local LLM setup tour will run on next launch\n"
            "• Model Manager hardware suggestions will be turned off\n\n"
            "Click Confirm to apply. You'll then be offered Restart now to run the tour immediately."
        ),
    ),
)


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
