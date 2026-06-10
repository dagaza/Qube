"""Sidecar companion line enqueue guard."""
from __future__ import annotations

import queue
import unittest

from workers.sidecar_llm_worker import SidecarLlmWorker


class TestCompanionVerbalEnqueue(unittest.TestCase):
    def test_enqueue_queues_when_sidecar_busy(self) -> None:
        worker = SidecarLlmWorker()
        worker.model_loaded = True
        worker._reloading = False
        worker._cmd_queue = queue.Queue()
        worker._cmd_queue.put({"op": "title"})
        self.assertTrue(worker.enqueue_companion_line({"trigger": "idle"}))
        ops = []
        while True:
            try:
                ops.append(worker._cmd_queue.get_nowait().get("op"))
            except queue.Empty:
                break
        self.assertEqual(ops, ["title", "companion_line"])

    def test_enqueue_accepts_when_queue_empty(self) -> None:
        worker = SidecarLlmWorker()
        worker.model_loaded = True
        worker._reloading = False
        worker._cmd_queue = queue.Queue()
        self.assertTrue(worker.enqueue_companion_line({"trigger": "idle"}))
        cmd = worker._cmd_queue.get_nowait()
        self.assertEqual(cmd.get("op"), "companion_line")


if __name__ == "__main__":
    unittest.main()
