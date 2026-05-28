"""Unit tests for reserved Library folder policy helpers."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.library_folder_policy import is_qube_managed_document_filename


class LibraryFolderPolicyTests(unittest.TestCase):
    def test_qube_managed_filename_heuristics(self) -> None:
        self.assertTrue(is_qube_managed_document_filename("qube/preferences.md"))
        self.assertTrue(is_qube_managed_document_filename("preferences.md"))
        self.assertTrue(is_qube_managed_document_filename("Qube knowledge.md"))
        self.assertFalse(is_qube_managed_document_filename("my-notes.pdf"))
        self.assertFalse(is_qube_managed_document_filename(""))


if __name__ == "__main__":
    unittest.main()
