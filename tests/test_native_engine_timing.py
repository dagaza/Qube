"""Tests for core.native_engine_timing."""
from __future__ import annotations

import time
import unittest

from core.native_engine_timing import EngineJobTiming, timing_from_cmd


class NativeEngineTimingTests(unittest.TestCase):
    def test_duration_fields_computed_in_ms(self) -> None:
        t0 = time.monotonic()
        timing = EngineJobTiming(
            submitted_at=t0,
            dequeued_at=t0 + 0.1,
            inference_started_at=t0 + 0.12,
            inference_finished_at=t0 + 0.15,
            finished_at=t0 + 0.16,
        )
        self.assertGreaterEqual(timing.queue_wait_ms, 90)
        self.assertGreaterEqual(timing.engine_prep_ms, 15)
        self.assertGreaterEqual(timing.inference_ms, 25)
        self.assertGreaterEqual(timing.total_ms, 150)

    def test_timing_from_cmd_round_trip(self) -> None:
        t0 = time.monotonic()
        cmd = {
            "request_id": "abc",
            "task": "chat",
            "debug_caller": "chat",
            "debug_exchange_id": 5,
            "priority_label": "interactive",
            "submitted_at": t0,
            "dequeued_at": t0 + 0.05,
            "inference_started_at": t0 + 0.06,
            "inference_finished_at": t0 + 0.08,
            "_timing_meta": {
                "queue_depth_at_submit": 2,
                "queue_depth_at_start": 1,
                "queued_behind": ["memory_extraction"],
            },
        }
        d = timing_from_cmd(cmd, finished_at=t0 + 0.09).to_dict()
        self.assertEqual(d["request_id"], "abc")
        self.assertEqual(d["queue_depth_at_submit"], 2)
        self.assertEqual(d["queued_behind"], ["memory_extraction"])
        self.assertGreater(d["queue_wait_ms"], 0)


if __name__ == "__main__":
    unittest.main()
