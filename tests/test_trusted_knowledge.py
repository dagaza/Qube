"""Tests for trusted knowledge pipeline and composer routing."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.composer_attachments import (  # noqa: E402
    ComposerAttachment,
    resolve_attachment_routing,
)
from core.knowledge.registry import resolve_turn_knowledge_service  # noqa: E402
from core.knowledge.types import (  # noqa: E402
    SERVICE_GENERAL_WEB,
    SERVICE_TRUSTED_KNOWLEDGE,
)
from core.knowledge.web_retrieval import run_v2_web_retrieval  # noqa: E402


_WIKI_ROW = {
    "title": "Bucharest",
    "snippet": "Bucharest is the capital and largest city of Romania.",
    "full_text": "Bucharest is the capital and largest city of Romania.",
    "url": "https://en.wikipedia.org/wiki/Bucharest",
    "pageid": 12345,
    "_wiki_source": True,
}


class TestTrustedKnowledge(unittest.TestCase):
    def test_composer_trusted_routing(self) -> None:
        patch = resolve_attachment_routing(
            [ComposerAttachment(kind="tool", id="trusted", label="Trusted")]
        )
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch["route"], "web")
        self.assertEqual(patch["strategy"], "attachment_tool_trusted")
        self.assertEqual(patch["attachment_tool"], "trusted")

    def test_resolve_turn_knowledge_service(self) -> None:
        self.assertEqual(
            resolve_turn_knowledge_service(composer_trusted=True),
            SERVICE_TRUSTED_KNOWLEDGE,
        )
        self.assertEqual(
            resolve_turn_knowledge_service(
                composer_internet=True, composer_trusted=True
            ),
            SERVICE_TRUSTED_KNOWLEDGE,
        )
        self.assertEqual(
            resolve_turn_knowledge_service(
                default_service=SERVICE_TRUSTED_KNOWLEDGE
            ),
            SERVICE_TRUSTED_KNOWLEDGE,
        )
        self.assertEqual(
            resolve_turn_knowledge_service(),
            SERVICE_GENERAL_WEB,
        )

    @patch("core.knowledge.pipeline_trusted.search_duckduckgo")
    @patch("core.knowledge.pipeline_trusted.search_wikipedia")
    def test_v2_trusted_service_bundle(self, mock_wiki, mock_ddg) -> None:
        mock_wiki.return_value = [_WIKI_ROW]
        mock_ddg.return_value = []

        outcome = run_v2_web_retrieval(
            query="What is the capital of Romania?",
            semantic_query="What is the capital of Romania?",
            knowledge_service=SERVICE_TRUSTED_KNOWLEDGE,
        )

        self.assertFalse(outcome.skip_enrichment)
        self.assertIsNotNone(outcome.bundle)
        assert outcome.bundle is not None
        self.assertEqual(outcome.bundle.knowledge_service, SERVICE_TRUSTED_KNOWLEDGE)
        self.assertIn("wikipedia_api", outcome.bundle.adapter_calls)
        self.assertEqual(len(outcome.bundle.sources), 1)
        self.assertEqual(outcome.bundle.sources[0].adapter, "wikipedia_api")
        self.assertEqual(outcome.bundle.sources[0].fetch_status, "abstract")
        self.assertGreater(outcome.bundle.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
