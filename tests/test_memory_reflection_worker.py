"""Memory reflection worker cadence and candidate priority."""
from __future__ import annotations

import unittest

from workers.memory_reflection_worker import (
    REFLECT_INTERVAL_IDLE_SEC,
    REFLECT_INTERVAL_SEC,
    reflection_candidate_priority,
)


class TestReflectionCandidatePriority(unittest.TestCase):
    def test_never_reflected_before_audited(self) -> None:
        never = reflection_candidate_priority({"last_reflected_at": 0})
        audited = reflection_candidate_priority({"last_reflected_at": 100.0})
        self.assertLess(never, audited)

    def test_unclear_before_other_labels_when_both_audited(self) -> None:
        unclear = reflection_candidate_priority(
            {"last_reflected_at": 200.0, "reflection_label": "unclear"}
        )
        durable = reflection_candidate_priority(
            {"last_reflected_at": 50.0, "reflection_label": "durable_user_fact"}
        )
        self.assertLess(unclear, durable)

    def test_older_audit_before_newer_when_same_label(self) -> None:
        older = reflection_candidate_priority(
            {"last_reflected_at": 10.0, "reflection_label": "durable_user_fact"}
        )
        newer = reflection_candidate_priority(
            {"last_reflected_at": 100.0, "reflection_label": "durable_user_fact"}
        )
        self.assertLess(older, newer)


class TestReflectionCadence(unittest.TestCase):
    def test_idle_interval_longer_than_active(self) -> None:
        self.assertGreater(REFLECT_INTERVAL_IDLE_SEC, REFLECT_INTERVAL_SEC)


if __name__ == "__main__":
    unittest.main()
