"""Tests for core.native_engine_queue.PriorityCommandQueue."""
from __future__ import annotations

import threading
import time
import unittest

from core.native_engine_queue import EnginePriority, PriorityCommandQueue, priority_label


class PriorityCommandQueueTests(unittest.TestCase):
    def test_interactive_dequeues_before_background(self) -> None:
        q = PriorityCommandQueue()
        q.put({"op": "chat_once", "debug_caller": "memory_extraction"}, priority=EnginePriority.background)
        q.put({"op": "generate", "debug_caller": "chat"}, priority=EnginePriority.interactive)
        first = q.get()
        self.assertEqual(first.get("op"), "generate")

    def test_fifo_within_same_priority(self) -> None:
        q = PriorityCommandQueue()
        q.put({"op": "chat_once", "n": 1}, priority=EnginePriority.background)
        q.put({"op": "chat_once", "n": 2}, priority=EnginePriority.background)
        first = q.get()
        second = q.get()
        self.assertEqual(first.get("n"), 1)
        self.assertEqual(second.get("n"), 2)

    def test_purge_removes_matching_entries(self) -> None:
        q = PriorityCommandQueue()
        q.put({"op": "chat_once"}, priority=EnginePriority.background)
        q.put({"op": "generate"}, priority=EnginePriority.interactive)
        removed = q.purge(lambda c: c.get("op") == "chat_once")
        self.assertEqual(removed, 1)
        self.assertEqual(q.get().get("op"), "generate")

    def test_snapshot_reports_depth_and_callers(self) -> None:
        q = PriorityCommandQueue()
        q.put({"op": "chat_once", "debug_caller": "memory_extraction"}, priority=EnginePriority.background)
        snap = q.snapshot()
        self.assertEqual(snap["depth_total"], 1)
        self.assertEqual(snap["depth_by_priority"].get("background"), 1)
        self.assertEqual(snap["queued_callers"], ["memory_extraction"])

    def test_stamps_submitted_at_and_dequeued_at(self) -> None:
        q = PriorityCommandQueue()
        stamped = q.put({"op": "generate"}, priority=EnginePriority.interactive)
        self.assertIn("submitted_at", stamped)
        self.assertIn("seq", stamped)
        cmd = q.get()
        self.assertIn("dequeued_at", cmd)
        self.assertGreaterEqual(cmd["dequeued_at"], stamped["submitted_at"])

    def test_get_timeout_raises_empty(self) -> None:
        q = PriorityCommandQueue()
        import queue as std_queue

        with self.assertRaises(std_queue.Empty):
            q.get(timeout=0.01)

    def test_priority_label(self) -> None:
        self.assertEqual(priority_label(EnginePriority.interactive), "interactive")


if __name__ == "__main__":
    unittest.main()
