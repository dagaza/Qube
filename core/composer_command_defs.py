"""Composer @-command definitions (PyQt-free — safe for build scripts)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComposerCommand:
    id: str
    label: str
    description: str
    requires_confirmation: bool = False
    confirmation_title: str = ""
    confirmation_message: str = ""


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
