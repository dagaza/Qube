"""Tests for RecipeExtractor (M6)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.discovery.duckduckgo import format_site_bias_query  # noqa: E402
from core.knowledge.extractors.recipe_extractor import (  # noqa: E402
    EXTRACTOR_NAME,
    RecipeExtractor,
)
from core.knowledge.extractors.registry import extract_document, select_best_extractor  # noqa: E402
from core.knowledge.site_bias import RECIPE_DEFAULT_SITE_BIAS  # noqa: E402
from core.knowledge.web_fetch_context import resolve_web_fetch_options  # noqa: E402
from core.knowledge.types import RetrievalContext  # noqa: E402

_FIXTURES = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "fetch"


class TestRecipeExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import trafilatura  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("trafilatura is not installed") from exc

    def test_supports_jsonld_recipe_fixture(self) -> None:
        html = (_FIXTURES / "recipe_jsonld.html").read_text(encoding="utf-8")
        extractor = RecipeExtractor()
        self.assertGreaterEqual(extractor.supports("https://example.com/recipe", html), 0.9)

    def test_extract_document_structured_ingredients(self) -> None:
        html = (_FIXTURES / "recipe_jsonld.html").read_text(encoding="utf-8")
        url = "https://example.com/recipes/carbonara"
        extractor, confidence = select_best_extractor(url, html)
        self.assertEqual(extractor.metadata.name, EXTRACTOR_NAME)
        self.assertGreaterEqual(confidence, 0.5)

        document = extract_document(html, url)
        self.assertIn("Carbonara", document.title or "")
        self.assertIn("ingredients", document.structured_data)
        ingredients = document.structured_data.get("ingredients") or []
        self.assertTrue(any("spaghetti" in str(item).lower() for item in ingredients))
        self.assertTrue(document.sections)

    @patch("core.knowledge.discovery.duckduckgo.search_duckduckgo_detailed")
    def test_recipe_site_bias_scopes_discovery_query(self, mock_search) -> None:
        mock_search.return_value = ([], None)
        from core.knowledge.discovery.duckduckgo import DuckDuckGoDiscovery

        DuckDuckGoDiscovery().discover(
            "carbonara recipe",
            max_results=3,
            site_bias=RECIPE_DEFAULT_SITE_BIAS[:2],
        )
        args, kwargs = mock_search.call_args
        query = args[0]
        self.assertIn("site:seriouseats.com", query)
        self.assertIn("site:bbcgoodfood.com", query)

    def test_format_site_bias_for_recipe_defaults(self) -> None:
        ctx = RetrievalContext(
            query="carbonara",
            semantic_query="carbonara",
            composer_tool="recipe",
        )
        options = resolve_web_fetch_options(ctx)
        query, target = format_site_bias_query("carbonara", options.site_bias)
        self.assertIsNone(target)
        self.assertIn("site:seriouseats.com", query)


if __name__ == "__main__":
    unittest.main()
