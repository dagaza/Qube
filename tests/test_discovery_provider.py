"""Tests for discovery providers (M1)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.discovery import discover  # noqa: E402
from core.knowledge.discovery.duckduckgo import (  # noqa: E402
    DuckDuckGoDiscovery,
    format_site_bias_query,
)
from core.knowledge.discovery.registry import (  # noqa: E402
    default_discovery_provider,
    list_discovery_providers,
)


class TestDiscoveryProvider(unittest.TestCase):
    def test_registry_lists_duckduckgo(self) -> None:
        providers = list_discovery_providers()
        self.assertIn("duckduckgo", providers)
        self.assertIn("wikipedia", providers)
        self.assertIn("brave_search", providers)
        self.assertEqual(default_discovery_provider().id, "duckduckgo")

    def test_format_site_bias_single_domain(self) -> None:
        query, target = format_site_bias_query("pasta recipe", ("seriouseats.com",))
        self.assertEqual(query, "pasta recipe")
        self.assertEqual(target, "seriouseats.com")

    def test_format_site_bias_multiple_domains(self) -> None:
        query, target = format_site_bias_query(
            "boss fight guide",
            ("fandom.com", "ign.com"),
        )
        self.assertEqual(target, None)
        self.assertIn("site:fandom.com", query)
        self.assertIn("site:ign.com", query)
        self.assertIn("boss fight guide", query)

    @patch("core.knowledge.discovery.duckduckgo.search_duckduckgo_detailed")
    def test_discover_returns_candidate_urls(self, mock_search) -> None:
        mock_search.return_value = (
            [
                {
                    "title": "Dust bathing",
                    "snippet": "Birds take dust baths.",
                    "url": "https://example.org/birds",
                },
                {
                    "title": "No URL row",
                    "snippet": "Should be skipped.",
                },
            ],
            {"response_kind": "serp", "http_status": 200, "parsed_rows": 2},
        )
        results = DuckDuckGoDiscovery().discover(
            "dust baths",
            max_results=5,
            site_bias=("example.org",),
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.org/birds")
        self.assertEqual(results[0].title, "Dust bathing")
        self.assertEqual(results[0].snippet, "Birds take dust baths.")
        self.assertEqual(results[0].source, "duckduckgo")
        self.assertEqual(results[0].rank, 0)
        mock_search.assert_called_once_with(
            "dust baths",
            max_results=5,
            target_site="example.org",
        )

    @patch("core.knowledge.discovery.duckduckgo.search_duckduckgo_detailed")
    def test_discover_module_helper(self, mock_search) -> None:
        mock_search.return_value = (
            [{"title": "A", "snippet": "B", "url": "https://a.test/page"}],
            {"response_kind": "serp", "http_status": 200, "parsed_rows": 1},
        )
        results = discover("query", max_results=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://a.test/page")


if __name__ == "__main__":
    unittest.main()
