"""Tests for v3 usage milestones."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.companion_cognition.usage_counters import (
    load_counters,
    record_session_start,
    should_emit_usage_pattern,
)


def _pick_milestone(data: dict) -> str | None:
    from core.companion_cognition import usage_counters as uc

    return uc._pick_milestone(data)


class TestCompanionMilestonesV3(unittest.TestCase):
    def test_days_milestone_priority(self) -> None:
        data = {
            "days_active": 30,
            "session_count": 100,
            "ingest_events": 100,
            "captions_emitted": 200,
            "milestones_emitted": ["days_7"],
        }
        self.assertEqual(_pick_milestone(data), "days_30")

    def test_usage_pattern_gate(self) -> None:
        data = {"days_active": 6, "last_usage_pattern_date": ""}
        self.assertFalse(should_emit_usage_pattern(data))
        data["days_active"] = 10
        self.assertTrue(should_emit_usage_pattern(data))

    def test_record_session_start_emits_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usage_counters.json"
            with patch("core.companion_cognition.usage_counters._COUNTERS_PATH", path):
                mid1, data = record_session_start()
                self.assertIsNone(mid1)
                data["days_active"] = 7
                path.write_text(json.dumps(data), encoding="utf-8")
                data2 = load_counters()
                data2["session_count"] = 10
                mid = _pick_milestone(data2)
                self.assertEqual(mid, "days_7")


if __name__ == "__main__":
    unittest.main()
