"""Regression: native no-model errors must persist as assistant turns in SQLite."""
from __future__ import annotations

import os
import re
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class NativeLoadErrorPersistenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.join(ROOT, "workers", "llm_worker.py")
        with open(path, "r", encoding="utf-8") as f:
            cls.src = f.read()

    def test_native_load_error_is_captured_for_persistence(self) -> None:
        self.assertIn("native_load_error_text", self.src)
        self.assertRegex(
            self.src,
            r'if\s+"native model not loaded"\s+in\s+err_txt\.lower\(\):'
            r'[\s\S]*?native_load_error_text\s*=\s*err_txt\.strip\(\)',
            "Expected the no-model error path to capture text for DB persistence.",
        )

    def test_empty_stream_falls_back_to_native_load_error_before_db_write(self) -> None:
        self.assertRegex(
            self.src,
            r'if\s+not\s+final_text\.strip\(\)\s+and\s+native_load_error_text:'
            r'[\s\S]*?final_text\s*=\s*native_load_error_text',
            "Expected empty native streams to persist the captured no-model error.",
        )

    def test_db_write_follows_final_text_assignment(self) -> None:
        assign_idx = self.src.find(
            "if not final_text.strip() and native_load_error_text:"
        )
        self.assertGreater(assign_idx, 0)
        persist_idx = self.src.find("_persist_assistant_turn(final_text", assign_idx)
        self.assertGreater(
            persist_idx,
            assign_idx,
            "Expected _persist_assistant_turn to run after the no-model fallback assignment.",
        )

    def test_native_load_error_skips_enrichment(self) -> None:
        self.assertRegex(
            self.src,
            r'if\s+"native model not loaded"\s+in\s+err_txt\.lower\(\):'
            r'[\s\S]*?_mark_skip_enrichment\(\s*"native_model_not_loaded"\s*\)',
            "Expected no-model errors to skip memory enrichment.",
        )


if __name__ == "__main__":
    unittest.main()
