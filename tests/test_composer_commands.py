"""Tests for composer @-command actions."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from core.composer_commands import COMPOSER_COMMANDS, execute_composer_command


class TestComposerCommands(unittest.TestCase):
    def test_reset_help_guidance_command_registered(self) -> None:
        ids = {c.id for c in COMPOSER_COMMANDS}
        self.assertIn("reset_help_guidance", ids)
        cmd = next(c for c in COMPOSER_COMMANDS if c.id == "reset_help_guidance")
        self.assertTrue(cmd.requires_confirmation)
        self.assertIn("Confirm", cmd.confirmation_message)

    @patch("core.composer_commands.reset_help_guidance_settings")
    def test_execute_reset_help_guidance(self, mock_reset: MagicMock) -> None:
        window = MagicMock()
        settings = MagicMock()
        cb = MagicMock()
        settings.model_manager_hardware_suggestions_cb = cb
        window.settings_view = settings
        window.model_manager_view = MagicMock()

        result = execute_composer_command("reset_help_guidance", window=window)

        mock_reset.assert_called_once()
        cb.blockSignals.assert_called()
        cb.setChecked.assert_called_with(False)
        window.model_manager_view.refresh_hardware_suggestions.assert_called_once()
        self.assertTrue(result.ok)
        self.assertFalse(result.dialog_message)
        self.assertIsNotNone(result.notification)
        assert result.notification is not None
        self.assertEqual(result.notification.action_id, "restart_app")
        self.assertIn("Restart", result.notification.action_label or "")

    def test_unknown_command(self) -> None:
        result = execute_composer_command("nope", window=None)
        self.assertFalse(result.ok)
        self.assertIn("Unknown", result.dialog_message)


if __name__ == "__main__":
    unittest.main()
