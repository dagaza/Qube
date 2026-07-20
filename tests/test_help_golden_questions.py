"""Phase 1 golden-question checks against generated reference content."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.help_corpus_manifest import bundled_help_locale_dir, load_manifest
from core.help_reference_generator import generate_all_reference_markdown

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "help_golden_questions.json"


class HelpGoldenQuestionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.doc_by_id = {
            str(doc["id"]): doc for doc in cls.manifest.get("documents") or []
        }
        cls.generated = generate_all_reference_markdown()
        cls.composed_root = bundled_help_locale_dir("en")

    def _doc_text(self, doc_id: str) -> str:
        doc = self.doc_by_id[doc_id]
        rel = str(doc["path"])
        if doc.get("generated"):
            return self.generated.get(rel, "")
        path = self.composed_root / rel
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def test_fixture_present(self) -> None:
        rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 30)

    def test_bundled_inventory_complete(self) -> None:
        manifest = load_manifest()
        doc_ids = {str(doc["id"]) for doc in manifest.get("documents") or []}
        required = {
            "index",
            "features.conversations",
            "features.library",
            "features.memory_manager",
            "features.model_manager",
            "features.telemetry",
            "release.whats_new",
            "release.migration_guide",
        }
        missing = required - doc_ids
        self.assertTrue(required.issubset(doc_ids), f"missing inventory ids: {missing}")
        self.assertGreaterEqual(len(manifest.get("canonical_answers") or []), 20)

    def test_golden_questions_map_to_manifest_and_content(self) -> None:
        rows = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(rows), 55)
        for row in rows:
            if row.get("negative"):
                continue
            for doc_id in row["expected_doc_ids"]:
                self.assertIn(doc_id, self.doc_by_id, row["question"])
                text = self._doc_text(doc_id)
                self.assertTrue(text, f"empty content for {doc_id}")
                for needle in row.get("must_mention") or []:
                    self.assertIn(
                        needle,
                        text,
                        f"{needle!r} missing for {doc_id} ({row['question']})",
                    )


if __name__ == "__main__":
    unittest.main()
