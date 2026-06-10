"""Tests for companion message library."""

from __future__ import annotations

import time
import unittest

from core.companion_cognition.message_library import (
    bundled_messages_path,
    load_message_library,
    validate_library_dict,
)
from core.companion_cognition.personality import vector_from_trait_preset
from core.companion_cognition.thoughts import think
from core.companion_cognition.types import CompanionObservation
from core.companion_cognition.variety import VarietySnapshot
from core.companion_verbal_traits import CompanionVerbalTraitPreset


class TestCompanionMessageLibrary(unittest.TestCase):
    def test_bundled_library_validates(self) -> None:
        import json

        data = json.loads(bundled_messages_path().read_text(encoding="utf-8"))
        ok, err = validate_library_dict(data)
        self.assertTrue(ok, err)
        self.assertGreaterEqual(len(data.get("messages") or []), 200)
        self.assertEqual(int(data.get("schema_version") or 0), 3)
        for msg in data.get("messages") or []:
            self.assertIn("voice", msg)
            self.assertTrue(msg.get("text"))

    def test_preview_bypasses_recent_ids(self) -> None:
        lib = load_message_library(bundled_messages_path())
        obs = CompanionObservation(type="settings_preview", facts={"daypart": "morning"})
        personality = vector_from_trait_preset(CompanionVerbalTraitPreset.WARM)
        variety = VarietySnapshot(
            now=time.time(),
            recent_message_ids=("prev_001", "prev_002"),
        )
        thought = think(obs, personality, variety)
        assert thought is not None
        msg = lib.select_message(thought, variety, personality, for_preview=True)
        self.assertIsNotNone(msg)

    def test_select_message_for_ingest(self) -> None:
        lib = load_message_library(bundled_messages_path())
        obs = CompanionObservation(
            type="library_update_completed",
            facts={"file_count": 1},
        )
        personality = vector_from_trait_preset(CompanionVerbalTraitPreset.WARM)
        variety = VarietySnapshot(now=time.time())
        thought = think(obs, personality, variety)
        assert thought is not None
        msg = lib.select_message(thought, variety, personality)
        self.assertIsNotNone(msg)
        assert msg is not None
        self.assertLessEqual(len(msg.text), 72)


if __name__ == "__main__":
    unittest.main()
