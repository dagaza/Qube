"""Tests for curated Model Manager hub catalog loading."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from core import qube_verified_models as qvm
from core.qube_verified_models import CatalogEntry


class TestQubeVerifiedModels(unittest.TestCase):
    def test_legacy_repo_id_becomes_gguf_repo(self) -> None:
        raw = [{"repo_id": "org/model-gguf", "title": "Display"}]
        entries = qvm._parse_catalog_raw(raw)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].gguf_repo, "org/model-gguf")
        self.assertEqual(entries[0].title, "Display")
        self.assertFalse(entries[0].is_catalog_card)

    def test_catalog_entry_parses_publisher_and_repos(self) -> None:
        raw = [
            {
                "id": "gemma-test",
                "title": "Gemma 4",
                "description": "MoE instruct",
                "publisher": "google",
                "gguf_repo": "unsloth/gemma-GGUF",
                "gguf_repos": [
                    "unsloth/gemma-GGUF",
                    "bartowski/gemma-GGUF",
                ],
            }
        ]
        entries = qvm._parse_catalog_raw(raw)
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e.catalog_id, "gemma-test")
        self.assertEqual(e.publisher, "google")
        self.assertEqual(e.gguf_repo, "unsloth/gemma-GGUF")
        self.assertEqual(e.gguf_repos, ("unsloth/gemma-GGUF", "bartowski/gemma-GGUF"))
        self.assertTrue(e.is_catalog_card)

    def test_branding_deepseek_ai_publisher(self) -> None:
        entry = CatalogEntry(
            catalog_id="ds",
            title="DeepSeek R1",
            description="",
            publisher="deepseek-ai",
            gguf_repo="deepseek-ai/DeepSeek-R1-GGUF",
        )
        with mock.patch.object(qvm, "_logo_exists", return_value=True):
            branding = qvm.branding_for_entry(entry, resolver=mock.Mock())
        self.assertIsNotNone(branding)
        assert branding is not None
        self.assertEqual(branding["name"], "DeepSeek")
        self.assertIn("deepseek.svg", branding["logo"])

    def test_branding_for_entry_uses_publisher_allowlist(self) -> None:
        entry = CatalogEntry(
            catalog_id="x",
            title="Gemma",
            description="",
            publisher="google",
            gguf_repo="unsloth/gemma-GGUF",
            gguf_repos=("unsloth/gemma-GGUF",),
        )
        with mock.patch.object(qvm, "_logo_exists", return_value=True):
            branding = qvm.branding_for_entry(entry, resolver=mock.Mock())
        self.assertIsNotNone(branding)
        assert branding is not None
        self.assertEqual(branding["name"], "Google")
        self.assertTrue(branding["official"])
        self.assertIn("Google.svg", branding["logo"])

    def test_branding_falls_back_to_resolver(self) -> None:
        entry = CatalogEntry(
            catalog_id="x",
            title="M",
            description="",
            publisher="",
            gguf_repo="bartowski/foo-GGUF",
        )
        resolver = mock.Mock()
        resolver.resolve_for_model.return_value = {"name": "HF", "logo": "/x", "official": False}
        branding = qvm.branding_for_entry(entry, resolver=resolver)
        resolver.resolve_for_model.assert_called_once_with("bartowski/foo-GGUF")
        self.assertEqual(branding, {"name": "HF", "logo": "/x", "official": False})

    def test_load_from_user_json(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            user = Path(td) / "qube_verified_models.json"
            user.write_text(
                json.dumps([{"repo_id": "acme/demo-gguf", "title": "Demo"}]),
                encoding="utf-8",
            )
            with (
                mock.patch.object(qvm, "ensure_user_verified_models_seeded", return_value=user),
                mock.patch.object(qvm, "_maybe_refresh_user_catalog_from_bundle"),
            ):
                entries = qvm.load_qube_verified_models()
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].gguf_repo, "acme/demo-gguf")
            self.assertEqual(entries[0].title, "Demo")

    def test_wrapped_models_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "wrapped.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [{"repo_id": "x/y", "title": "Y"}],
                    }
                ),
                encoding="utf-8",
            )
            entries = qvm._read_catalog_file(path)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].gguf_repo, "x/y")

    def test_unwrap_double_wrapped_catalog(self) -> None:
        raw = [
            {
                "schema_version": 1,
                "models": [{"repo_id": "a/b", "title": "B"}],
            }
        ]
        entries = qvm._parse_catalog_raw(raw)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].gguf_repo, "a/b")

    def test_refresh_when_bundled_schema_newer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bundled = td_path / "bundled.json"
            user = td_path / "user.json"
            bundled.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "models": [
                            {"id": "a", "gguf_repo": "a/one", "title": "One"},
                            {"id": "b", "gguf_repo": "b/two", "title": "Two"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            user.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [{"id": "a", "gguf_repo": "a/one", "title": "One"}],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(qvm, "bundled_verified_models_path", return_value=bundled):
                qvm._maybe_refresh_user_catalog_from_bundle(user)
            self.assertEqual(len(qvm._read_catalog_file(user)), 2)

    def test_refresh_when_user_catalog_is_subset_of_bundled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bundled = td_path / "bundled.json"
            user = td_path / "user.json"
            bundled.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {"id": "a", "gguf_repo": "a/one", "title": "One"},
                            {"id": "b", "gguf_repo": "b/two", "title": "Two"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            user.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [{"id": "a", "gguf_repo": "a/one", "title": "One"}],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(qvm, "bundled_verified_models_path", return_value=bundled):
                qvm._maybe_refresh_user_catalog_from_bundle(user)
            self.assertEqual(len(qvm._read_catalog_file(user)), 2)

    def test_refresh_when_bundled_mtime_newer_than_user(self) -> None:
        """Editing assets/ bumps mtime; ~/.qube should be replaced on next load."""
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bundled = td_path / "bundled.json"
            user = td_path / "user.json"
            user.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "models": [{"id": "a", "gguf_repo": "a/x", "title": "Only"}],
                    }
                ),
                encoding="utf-8",
            )
            time.sleep(0.05)
            bundled.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "models": [
                            {"id": "a", "gguf_repo": "a/x", "title": "Only"},
                            {"id": "b", "gguf_repo": "b/y", "title": "Added"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(qvm, "bundled_verified_models_path", return_value=bundled):
                qvm._maybe_refresh_user_catalog_from_bundle(user)
            self.assertEqual(len(qvm._read_catalog_file(user)), 2)

    def test_maybe_refresh_legacy_user_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            bundled = td_path / "bundled.json"
            user = td_path / "user.json"
            bundled.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": [
                            {"gguf_repo": "a/one", "title": "One"},
                            {"gguf_repo": "b/two", "title": "Two"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            user.write_text(
                json.dumps([{"repo_id": "a/one", "title": "One"}]),
                encoding="utf-8",
            )
            with mock.patch.object(qvm, "bundled_verified_models_path", return_value=bundled):
                qvm._maybe_refresh_user_catalog_from_bundle(user)
            refreshed = qvm._read_catalog_file(user)
            self.assertEqual(len(refreshed), 2)

    def test_normalize_entries_legacy_compat(self) -> None:
        raw = [
            {"repo_id": "org/model", "title": "Display"},
            {"title": "No repo"},
        ]
        out = qvm._normalize_entries(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["repo_id"], "org/model")


if __name__ == "__main__":
    unittest.main()
