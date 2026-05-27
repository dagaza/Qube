"""Tests for LanceDB row identity helpers."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.lance_row_id import lance_row_delete_filter, lance_row_id


class LanceRowIdTests(unittest.TestCase):
    def test_prefers_rowid(self):
        self.assertEqual(lance_row_id({"_rowid": 42, "id": "legacy"}), "42")

    def test_falls_back_to_legacy_id(self):
        self.assertEqual(lance_row_id({"id": "row-1"}), "row-1")

    def test_delete_filter_for_rowid(self):
        self.assertEqual(lance_row_delete_filter("17"), "_rowid = 17")

    def test_delete_filter_for_legacy_uuid(self):
        self.assertEqual(
            lance_row_delete_filter("abc-def"),
            "id = 'abc-def'",
        )


if __name__ == "__main__":
    unittest.main()
