"""Sidecar priority queue and burst-cap helpers."""
from __future__ import annotations

import time
import unittest

from core.sidecar_engine_queue import (
    COMPANION_DEFER_QUEUE_DEPTH,
    INGEST_BLURB_MAX_QUEUED,
    SidecarCommandQueue,
    SidecarPriority,
    priority_for_sidecar_cmd,
    should_defer_companion_line,
    should_drop_ingest_blurb,
)
from core.sidecar_types import SidecarTask


class TestSidecarPriority(unittest.TestCase):
    def test_foreground_tasks_are_interactive(self) -> None:
        for task in (SidecarTask.query_rewrite, SidecarTask.source_digest):
            pri = priority_for_sidecar_cmd({"op": "task", "task": task})
            self.assertEqual(pri, SidecarPriority.interactive)

    def test_background_ops(self) -> None:
        pri = priority_for_sidecar_cmd({"op": "title"})
        self.assertEqual(pri, SidecarPriority.background)
        pri_reload = priority_for_sidecar_cmd({"op": "reload"})
        self.assertEqual(pri_reload, SidecarPriority.control)

    def test_interactive_jumps_ahead_of_background(self) -> None:
        q = SidecarCommandQueue()
        q.put({"op": "title", "label": "bg1"})
        q.put({"op": "ingest_blurb", "filename": "a.txt"})
        q.put(
            {
                "op": "task",
                "task": SidecarTask.query_rewrite,
                "label": "fg",
            }
        )
        first = q.get(timeout=0.5)
        self.assertEqual(first.get("label"), "fg")
        self.assertEqual(first.get("task"), SidecarTask.query_rewrite)

    def test_fifo_within_same_priority(self) -> None:
        q = SidecarCommandQueue()
        q.put({"op": "title", "n": 1})
        q.put({"op": "title", "n": 2})
        a = q.get(timeout=0.5)
        b = q.get(timeout=0.5)
        self.assertEqual(a.get("n"), 1)
        self.assertEqual(b.get("n"), 2)

    def test_queue_wait_stamped(self) -> None:
        q = SidecarCommandQueue()
        q.put({"op": "title"})
        time.sleep(0.02)
        cmd = q.get(timeout=0.5)
        self.assertIn("submitted_at", cmd)
        self.assertIn("dequeued_at", cmd)
        self.assertGreaterEqual(float(cmd["dequeued_at"]) - float(cmd["submitted_at"]), 0.0)


class TestSidecarBurstCaps(unittest.TestCase):
    def test_companion_defer_depth_constant(self) -> None:
        self.assertGreaterEqual(COMPANION_DEFER_QUEUE_DEPTH, 4)

    def test_ingest_blurb_coalesce(self) -> None:
        q = SidecarCommandQueue()
        q.put({"op": "ingest_blurb", "filename": "doc.pdf", "sample_text": "a"})
        q.put({"op": "ingest_blurb", "filename": "doc.pdf", "sample_text": "b"})
        removed = q.purge(
            lambda c: c.get("op") == "ingest_blurb" and c.get("filename") == "doc.pdf"
        )
        self.assertEqual(removed, 2)
        q.put({"op": "ingest_blurb", "filename": "doc.pdf", "sample_text": "c"})
        self.assertEqual(q.qsize(), 1)
        cmd = q.get(timeout=0.5)
        self.assertEqual(cmd.get("sample_text"), "c")


class TestBurstPolicyHelpers(unittest.TestCase):
    def test_companion_defer_at_depth_cap(self) -> None:
        self.assertFalse(should_defer_companion_line(COMPANION_DEFER_QUEUE_DEPTH - 1))
        self.assertTrue(should_defer_companion_line(COMPANION_DEFER_QUEUE_DEPTH))

    def test_ingest_drop_at_cap(self) -> None:
        self.assertFalse(should_drop_ingest_blurb(INGEST_BLURB_MAX_QUEUED - 1))
        self.assertTrue(should_drop_ingest_blurb(INGEST_BLURB_MAX_QUEUED))


if __name__ == "__main__":
    unittest.main()
