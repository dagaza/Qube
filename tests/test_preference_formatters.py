"""Tests for preference-driven web snippet formatting."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.preference_formatters import format_web_snippets
from core.preference_policy import PreferenceField, PreferencePolicy


class PreferenceFormatterTests(unittest.TestCase):
    def test_metric_appends_celsius_hint(self):
        policy = PreferencePolicy(
            fields={"units": PreferenceField("metric", "explicit")},
        )
        rows = [{"title": "Weather", "snippet": "High 72°F with winds 10 mph."}]
        out = format_web_snippets(rows, policy)
        self.assertIn("°C", out[0]["snippet"])
        self.assertIn("km/h", out[0]["snippet"])

    def test_no_units_leaves_snippet_unchanged(self):
        policy = PreferencePolicy()
        rows = [{"title": "Weather", "snippet": "High 72°F."}]
        out = format_web_snippets(rows, policy)
        self.assertEqual(out[0]["snippet"], "High 72°F.")


if __name__ == "__main__":
    unittest.main()
