"""Tests for deep-research UI helpers."""

import unittest

from core.knowledge.deep_research_ui import (
    deep_research_available,
    deep_research_progress_percent,
)


class TestDeepResearchUi(unittest.TestCase):
    def test_progress_percent_phases(self) -> None:
        self.assertEqual(
            deep_research_progress_percent({"phase": "decomposing"}),
            12,
        )
        self.assertEqual(
            deep_research_progress_percent(
                {"phase": "retrieving", "sub_query_index": 2, "sub_query_total": 3}
            ),
            15 + int(55 * 2 / 3),
        )
        self.assertEqual(deep_research_progress_percent({"phase": "merging"}), 85)
        self.assertEqual(deep_research_progress_percent({"phase": "reporting"}), 95)
        self.assertEqual(deep_research_progress_percent({"phase": "synthesizing"}), 92)

    def test_availability_gate(self) -> None:
        self.assertFalse(
            deep_research_available(enabled=False, external_v2=True),
        )
        self.assertFalse(
            deep_research_available(enabled=True, external_v2=False),
        )
        self.assertTrue(
            deep_research_available(enabled=True, external_v2=True),
        )


if __name__ == "__main__":
    unittest.main()
