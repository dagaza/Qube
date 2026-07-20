"""Tests for bundled help corpus manifest validation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.help_corpus_manifest import (
    bundled_help_manifest_path,
    help_doc_source,
    load_manifest,
    manifest_is_compatible_with_app,
    validate_help_manifest,
)


class HelpManifestTests(unittest.TestCase):
    def test_bundled_manifest_validates(self) -> None:
        manifest = load_manifest()
        ok, err = validate_help_manifest(manifest)
        self.assertTrue(ok, err)

    def test_help_doc_source_prefix(self) -> None:
        self.assertEqual(
            help_doc_source("features/settings/knowledge.md"),
            "qube/documentation/features/settings/knowledge.md",
        )

    def test_manifest_rejects_missing_composed_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "00-index.md").write_text("# Index\n", encoding="utf-8")
            manifest = {
                "locale": "en",
                "corpus_version": "0.0.1",
                "collection_id": "qube.documentation",
                "documents": [
                    {
                        "id": "index",
                        "path": "00-index.md",
                        "title": "Index",
                        "type": "index",
                    },
                    {
                        "id": "missing",
                        "path": "missing.md",
                        "title": "Missing",
                        "type": "feature",
                    },
                ],
            }
            ok, err = validate_help_manifest(manifest, composed_root=root)
            self.assertFalse(ok)
            self.assertIn("missing composed document", err)

    def test_manifest_rejects_orphan_composed_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "00-index.md").write_text("# Index\n", encoding="utf-8")
            (root / "orphan.md").write_text("# Orphan\n", encoding="utf-8")
            manifest = {
                "locale": "en",
                "corpus_version": "0.0.1",
                "collection_id": "qube.documentation",
                "documents": [
                    {
                        "id": "index",
                        "path": "00-index.md",
                        "title": "Index",
                        "type": "index",
                    }
                ],
            }
            ok, err = validate_help_manifest(manifest, composed_root=root)
        self.assertFalse(ok)
        self.assertIn("orphan composed markdown", err)

    def test_manifest_rejects_inline_generated_markers_in_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()
            (root / "00-index.md").write_text("# Index\n", encoding="utf-8")
            (source / "bad.md").write_text(
                "# Bad\n<!-- GENERATED BEGIN controls -->\n",
                encoding="utf-8",
            )
            manifest = {
                "locale": "en",
                "corpus_version": "0.0.1",
                "collection_id": "qube.documentation",
                "documents": [
                    {
                        "id": "index",
                        "path": "00-index.md",
                        "title": "Index",
                        "type": "index",
                    }
                ],
            }
            ok, err = validate_help_manifest(manifest, composed_root=root)
            self.assertFalse(ok)
            self.assertIn("inline GENERATED markers", err)

    def test_manifest_is_compatible_with_app(self) -> None:
        manifest = load_manifest(bundled_help_manifest_path())
        ok, _ = manifest_is_compatible_with_app(manifest, "1.0.1")
        self.assertTrue(ok)

        blocked = dict(manifest)
        blocked["min_app_version"] = "99.0.0"
        ok, err = manifest_is_compatible_with_app(blocked, "1.0.1")
        self.assertFalse(ok)
        self.assertIn("min_app_version", err)


if __name__ == "__main__":
    unittest.main()
