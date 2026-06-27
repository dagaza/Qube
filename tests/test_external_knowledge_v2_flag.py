"""Tests for external knowledge v2 feature flag."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.app_settings import (  # noqa: E402
    external_knowledge_v2_enabled,
    get_external_knowledge_v2_enabled,
)


class TestExternalKnowledgeV2Flag(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("QUBE_EXTERNAL_KNOWLEDGE_V2", None)

    @patch("core.app_settings._store")
    def test_default_off(self, mock_store) -> None:
        mock_store.return_value.get.return_value = False
        self.assertFalse(get_external_knowledge_v2_enabled())
        self.assertFalse(external_knowledge_v2_enabled())

    @patch("core.app_settings._store")
    def test_settings_toggle(self, mock_store) -> None:
        mock_store.return_value.get.return_value = True
        self.assertTrue(get_external_knowledge_v2_enabled())

    @patch("core.app_settings._store")
    @patch.dict(os.environ, {"QUBE_EXTERNAL_KNOWLEDGE_V2": "1"})
    def test_env_override_on(self, mock_store) -> None:
        mock_store.return_value.get.return_value = False
        self.assertTrue(external_knowledge_v2_enabled())


if __name__ == "__main__":
    unittest.main()
