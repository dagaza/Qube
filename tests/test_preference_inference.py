"""Tests for conversational preference inference."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.preference_inference import (
    PREFERENCE_KIND_PRESENTATION,
    apply_inferred_to_profile,
    classify_preference_from_fact,
    infer_from_text,
)
from core.user_profile import UserProfileStore


class PreferenceInferenceTests(unittest.TestCase):
    def test_metric_units_phrase(self):
        inferred = infer_from_text("I always use metric units for measurements.")
        self.assertIsNotNone(inferred)
        assert inferred is not None
        self.assertEqual(inferred.profile_key, "units")
        self.assertEqual(inferred.value, "metric")

    def test_classify_presentation_fact(self):
        fact = {
            "content": "The user prefers metric units.",
            "category": "preference",
            "subject": "user",
            "origin": "user_stated",
        }
        kind, key = classify_preference_from_fact(fact)
        self.assertEqual(kind, PREFERENCE_KIND_PRESENTATION)
        self.assertEqual(key, "units")

    def test_apply_inferred_writes_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "user_profile.json")
            store = UserProfileStore(path=path)
            with patch("core.preference_inference.get_user_profile_store", return_value=store):
                fact = {
                    "content": "I prefer metric units.",
                    "category": "preference",
                    "subject": "user",
                    "origin": "user_stated",
                    "confidence": 0.9,
                }
                applied = apply_inferred_to_profile(fact)
                self.assertIsNotNone(applied)
                prefs = store.get_inferred_preferences()
                self.assertEqual(prefs["units"]["value"], "metric")


if __name__ == "__main__":
    unittest.main()
