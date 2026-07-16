"""Tests for web fetch context resolution (M5)."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.retrieval_profiles import (  # noqa: E402
    PROFILE_BALANCED,
    PROFILE_FAST,
    get_profile_spec,
)
from core.knowledge.site_bias import RECIPE_DEFAULT_SITE_BIAS  # noqa: E402
from core.knowledge.types import RetrievalContext  # noqa: E402
from core.knowledge.web_fetch_context import (  # noqa: E402
    resolve_web_fetch_options,
    resolve_web_relevance_options,
)


class TestWebFetchContext(unittest.TestCase):
    def test_fast_profile_has_zero_fetch(self) -> None:
        self.assertEqual(get_profile_spec(PROFILE_FAST).fetch_url_count, 0)

    def test_balanced_profile_fetches_one(self) -> None:
        self.assertEqual(get_profile_spec(PROFILE_BALANCED).fetch_url_count, 1)

    def test_fetch_pin_forces_fetch_on_fast_profile(self) -> None:
        ctx = RetrievalContext(
            query="guide",
            semantic_query="guide",
            retrieval_profile="fast",
            composer_tool="fetch",
        )
        options = resolve_web_fetch_options(ctx)
        self.assertGreaterEqual(options.fetch_url_count, 1)

    def test_recipe_pin_applies_site_bias(self) -> None:
        ctx = RetrievalContext(
            query="carbonara",
            semantic_query="carbonara",
            retrieval_profile="balanced",
            composer_tool="recipe",
        )
        options = resolve_web_fetch_options(ctx)
        self.assertGreaterEqual(options.fetch_url_count, 1)
        self.assertEqual(options.site_bias, RECIPE_DEFAULT_SITE_BIAS)

    def test_recipe_fetch_skips_relevance_gate(self) -> None:
        ctx = RetrievalContext(
            query="spaghetti carbonara recipe",
            semantic_query="spaghetti carbonara recipe",
            retrieval_profile="balanced",
            composer_tool="recipe",
        )
        fetch_options = resolve_web_fetch_options(ctx)
        relevance = resolve_web_relevance_options(ctx, fetch_options)
        self.assertFalse(relevance.apply_gate)
        self.assertEqual(relevance.mode, "recipe_fetch_skip")

    def test_fetch_pin_uses_permissive_gate(self) -> None:
        ctx = RetrievalContext(
            query="birds dust baths",
            semantic_query="birds dust baths",
            retrieval_profile="fast",
            composer_tool="fetch",
        )
        fetch_options = resolve_web_fetch_options(ctx)
        relevance = resolve_web_relevance_options(ctx, fetch_options)
        self.assertTrue(relevance.apply_gate)
        self.assertEqual(relevance.mode, "fetch_permissive")
        self.assertFalse(relevance.use_embedding_gate)
        self.assertEqual(relevance.min_token_ratio, 0.08)

    def test_general_web_stays_strict(self) -> None:
        ctx = RetrievalContext(
            query="birds",
            semantic_query="birds",
            retrieval_profile="balanced",
        )
        fetch_options = resolve_web_fetch_options(ctx)
        relevance = resolve_web_relevance_options(ctx, fetch_options)
        self.assertTrue(relevance.apply_gate)
        self.assertEqual(relevance.mode, "strict")
        self.assertTrue(relevance.use_embedding_gate)


if __name__ == "__main__":
    unittest.main()
