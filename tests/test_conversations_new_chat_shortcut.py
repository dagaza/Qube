"""Window shortcut for starting a new conversation on the Conversations page."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from ui.main_window import MAIN_STAGE_CONVERSATIONS, MainWindow


class _ShortcutHost:
    def __init__(self, *, stage_index: int, conversations_view=None) -> None:
        self._stage_index = stage_index
        self.conversations_view = conversations_view

    def _active_main_stage_index(self) -> int:
        return self._stage_index


class TestConversationsNewChatShortcut(unittest.TestCase):

    def test_shortcut_starts_new_chat_on_conversations_stage(self) -> None:
        cv = MagicMock()
        cv.focus_composer_if_ready = MagicMock()
        host = _ShortcutHost(
            stage_index=MAIN_STAGE_CONVERSATIONS, conversations_view=cv
        )

        with patch("ui.main_window.QTimer.singleShot") as single_shot:
            MainWindow._on_new_conversation_shortcut(host)

        cv._start_new_chat.assert_called_once_with()
        single_shot.assert_called_once_with(0, cv.focus_composer_if_ready)

    def test_shortcut_ignored_on_other_main_stages(self) -> None:
        cv = MagicMock()
        host = _ShortcutHost(
            stage_index=MAIN_STAGE_CONVERSATIONS + 1, conversations_view=cv
        )

        MainWindow._on_new_conversation_shortcut(host)

        cv._start_new_chat.assert_not_called()


if __name__ == "__main__":
    unittest.main()
