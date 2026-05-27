"""Tests for web-veto fallback prompt wiring."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.memory_filters import (
    PREFERENCE_APPLICATION_SUFFIX,
    WEB_CAPABILITY_DISABLED_SUFFIX,
)
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt


class WebVetoFallbackTests(unittest.TestCase):
    def test_web_capability_blocked_suffix(self):
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            web_capability_blocked=True,
        )
        system = compose_system_prompt(blocks)
        self.assertIn(WEB_CAPABILITY_DISABLED_SUFFIX.strip()[:40], system)

    def test_preference_suffix_when_requested(self):
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=False,
            apply_preference_suffix=True,
            preference_context="User prefs: metric units. Apply silently.",
        )
        system = compose_system_prompt(blocks)
        self.assertIn(PREFERENCE_APPLICATION_SUFFIX.strip()[:30], system)
        self.assertIn("metric units", system)

    def test_explicit_remember_skips_preference_suffix(self):
        blocks = build_prompt_blocks(
            execution_route="NONE",
            explicit_remember_active=True,
            explicit_remember_body="remember I like tea",
            apply_preference_suffix=True,
            preference_context="User prefs: metric units.",
        )
        system = compose_system_prompt(blocks)
        self.assertNotIn(PREFERENCE_APPLICATION_SUFFIX.strip()[:30], system)


if __name__ == "__main__":
    unittest.main()
