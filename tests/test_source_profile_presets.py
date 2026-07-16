"""Tests for general_web source profiles (M8)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.presets import (  # noqa: E402
    KnowledgePreset,
    load_preset,
    normalize_site_bias,
    preset_retrieval_overrides,
    save_preset,
)
from core.knowledge.registry import (  # noqa: E402
    adapter_filter_for_composer_tool,
    resolve_preset_retrieval_overrides,
    resolve_turn_knowledge_service,
)
from core.knowledge.types import (  # noqa: E402
    SERVICE_GENERAL_WEB,
    SERVICE_PRESET_KNOWLEDGE,
    RetrievalContext,
)
from core.knowledge.web_fetch_context import resolve_web_fetch_options  # noqa: E402


class TestSourceProfilePresets(unittest.TestCase):
    def test_normalize_site_bias_strips_protocol_and_www(self) -> None:
        self.assertEqual(
            normalize_site_bias(
                ["https://www.seriouseats.com/recipes", "seriouseats.com"]
            ),
            ["seriouseats.com"],
        )

    def test_general_web_preset_roundtrip(self) -> None:
        with patch(
            "core.knowledge.presets.user_data_root",
            return_value=self._tmpdir(),
        ) as root_patch:
            root = root_patch.return_value
            root.mkdir(parents=True, exist_ok=True)
            preset = KnowledgePreset(
                id="serious-eats",
                label="My Recipes",
                base_service=SERVICE_GENERAL_WEB,
                site_bias=["seriouseats.com"],
                fetch_url_count=2,
            )
            save_preset(preset)
            loaded = load_preset("serious-eats")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.base_service, SERVICE_GENERAL_WEB)
            self.assertEqual(loaded.site_bias, ["seriouseats.com"])
            self.assertEqual(loaded.fetch_url_count, 2)

    def test_general_web_preset_routes_to_general_web_service(self) -> None:
        with patch("core.knowledge.presets.user_data_root", return_value=self._tmpdir()):
            save_preset(
                KnowledgePreset(
                    id="serious-eats",
                    label="My Recipes",
                    base_service=SERVICE_GENERAL_WEB,
                    site_bias=["seriouseats.com"],
                    fetch_url_count=2,
                )
            )
            service = resolve_turn_knowledge_service(composer_tool="user:serious-eats")
            self.assertEqual(service, SERVICE_GENERAL_WEB)
            self.assertIsNone(adapter_filter_for_composer_tool("user:serious-eats"))

    def test_api_preset_still_routes_to_preset_knowledge(self) -> None:
        with patch("core.knowledge.presets.user_data_root", return_value=self._tmpdir()):
            save_preset(
                KnowledgePreset(
                    id="biology",
                    label="Biology",
                    adapters=["pubmed"],
                )
            )
            service = resolve_turn_knowledge_service(composer_tool="user:biology")
            self.assertEqual(service, SERVICE_PRESET_KNOWLEDGE)
            self.assertEqual(adapter_filter_for_composer_tool("user:biology"), ("pubmed",))

    def test_preset_retrieval_overrides(self) -> None:
        with patch("core.knowledge.presets.user_data_root", return_value=self._tmpdir()):
            save_preset(
                KnowledgePreset(
                    id="serious-eats",
                    label="My Recipes",
                    base_service=SERVICE_GENERAL_WEB,
                    site_bias=["seriouseats.com", "bbcgoodfood.com"],
                    fetch_url_count=2,
                )
            )
            overrides = resolve_preset_retrieval_overrides("user:serious-eats")
            self.assertEqual(
                overrides,
                {
                    "site_bias": ("seriouseats.com", "bbcgoodfood.com"),
                    "fetch_url_count": 2,
                },
            )
            self.assertEqual(preset_retrieval_overrides("biology"), {})

    def test_web_fetch_context_uses_preset_site_bias(self) -> None:
        ctx = RetrievalContext(
            query="carbonara",
            semantic_query="carbonara",
            retrieval_profile="balanced",
            composer_tool="user:serious-eats",
            site_bias=("seriouseats.com",),
            fetch_url_count=2,
        )
        options = resolve_web_fetch_options(ctx)
        self.assertEqual(options.site_bias, ("seriouseats.com",))
        self.assertEqual(options.fetch_url_count, 2)

    def test_reserved_preset_ids_include_fetch_and_recipe(self) -> None:
        from core.knowledge.presets import RESERVED_PRESET_IDS

        self.assertIn("fetch", RESERVED_PRESET_IDS)
        self.assertIn("recipe", RESERVED_PRESET_IDS)

    def _tmpdir(self):
        import tempfile
        from pathlib import Path

        return Path(tempfile.mkdtemp())


if __name__ == "__main__":
    unittest.main()
