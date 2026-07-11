"""Sticky Web toggle: worker flag persists until explicitly cleared."""

import unittest
from unittest.mock import patch

from workers.llm_worker import LLMWorker


class WebToggleStickyTests(unittest.TestCase):
    @patch.object(LLMWorker, "__init__", lambda self, *a, **k: None)
    def _bare_worker(self) -> LLMWorker:
        worker = LLMWorker.__new__(LLMWorker)
        worker._force_web_enabled = False
        return worker

    def test_force_web_enabled_persists_until_cleared(self) -> None:
        worker = self._bare_worker()
        worker.set_force_web_enabled(True)
        self.assertTrue(worker._force_web_enabled)
        self.assertTrue(worker._force_web_enabled)
        worker.set_force_web_enabled(False)
        self.assertFalse(worker._force_web_enabled)

    def test_set_force_web_next_turn_is_alias(self) -> None:
        worker = self._bare_worker()
        worker.set_force_web_next_turn(True)
        self.assertTrue(worker._force_web_enabled)
        worker.set_force_web_next_turn(False)
        self.assertFalse(worker._force_web_enabled)

    def test_web_search_active_signal_declared(self) -> None:
        self.assertTrue(hasattr(LLMWorker, "web_search_active"))


if __name__ == "__main__":
    unittest.main()
