"""Tests for INSPECT routing explainability text."""

from __future__ import annotations

import unittest

from core.knowledge.routing_inspect_explain import format_routing_inspect_text


class RoutingInspectExplainTests(unittest.TestCase):
    def test_empty_returns_blank(self) -> None:
        self.assertEqual(format_routing_inspect_text(None), "")

    def test_formats_route_and_hits(self) -> None:
        text = format_routing_inspect_text(
            {
                "route": "none",
                "route_pre_policy": "rag",
                "strategy": "adaptive_v4",
                "top_intent": "rag",
                "top_score": 0.42,
                "summary": "Empty library after gate",
                "trace": {
                    "retrieval_outcome": {
                        "downgrade_fired": True,
                        "memory_hits": 0,
                        "rag_hits": 0,
                        "web_hits": 0,
                    }
                },
            }
        )
        self.assertIn("Routing (this turn)", text)
        self.assertIn("rag → none", text)
        self.assertIn("Empty-source downgrade", text)
        self.assertIn("Routing debug log", text)


if __name__ == "__main__":
    unittest.main()
