"""Tests for preference policy merge and tool augmentation."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.preference_policy import apply_tool_policy, resolve_preference_policy
from core.user_profile import UserProfileStore


class PreferencePolicyMergeTests(unittest.TestCase):
    def test_explicit_overrides_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = os.path.join(tmp, "user_profile.json")
            store = UserProfileStore(path=profile_path)
            store.set_inferred("units", "imperial", confidence=0.9)
            with patch("core.preference_policy.get_user_profile_store", return_value=store):
                with patch("core.preference_policy.get_profile_units", return_value="metric"):
                    with patch("core.preference_policy.get_profile_locale", return_value=None):
                        with patch("core.preference_policy.get_profile_display_name", return_value=None):
                            with patch("core.preference_policy.get_profile_verbosity", return_value=None):
                                policy = resolve_preference_policy()
            self.assertEqual(policy.get("units"), "metric")
            self.assertEqual(policy.provenance_of("units"), "explicit")

    def test_inferred_fills_when_explicit_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = os.path.join(tmp, "user_profile.json")
            store = UserProfileStore(path=profile_path)
            store.set_inferred("units", "metric", confidence=0.85)
            with patch("core.preference_policy.get_user_profile_store", return_value=store):
                with patch("core.preference_policy.get_profile_units", return_value=None):
                    with patch("core.preference_policy.get_profile_locale", return_value=None):
                        with patch("core.preference_policy.get_profile_display_name", return_value=None):
                            with patch("core.preference_policy.get_profile_verbosity", return_value=None):
                                policy = resolve_preference_policy()
            self.assertEqual(policy.get("units"), "metric")
            self.assertEqual(policy.provenance_of("units"), "inferred")

    def test_session_overrides_explicit(self):
        with patch("core.preference_policy.get_user_profile_store") as mock_store:
            mock_store.return_value.get_inferred_preferences.return_value = {}
            with patch("core.preference_policy.get_profile_units", return_value="metric"):
                with patch("core.preference_policy.get_profile_locale", return_value=None):
                    with patch("core.preference_policy.get_profile_display_name", return_value=None):
                        with patch("core.preference_policy.get_profile_verbosity", return_value=None):
                            policy = resolve_preference_policy(
                                session_overrides={"units": "imperial"},
                            )
        self.assertEqual(policy.get("units"), "imperial")
        self.assertEqual(policy.provenance_of("units"), "session")

    def test_tool_policy_augments_weather_query(self):
        with patch("core.preference_policy.get_user_profile_store") as mock_store:
            mock_store.return_value.get_inferred_preferences.return_value = {
                "units": {"value": "metric", "confidence": 0.9, "source": "conversation"},
            }
            with patch("core.preference_policy.get_profile_units", return_value=None):
                with patch("core.preference_policy.get_profile_locale", return_value=None):
                    with patch("core.preference_policy.get_profile_display_name", return_value=None):
                        with patch("core.preference_policy.get_profile_verbosity", return_value=None):
                            policy = resolve_preference_policy()
        out = apply_tool_policy("What's the weather like?", policy, tool="internet")
        self.assertIn("metric", out.lower())
        self.assertIn("celsius", out.lower())


if __name__ == "__main__":
    unittest.main()
