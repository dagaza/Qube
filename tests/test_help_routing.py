"""Tests for @help routing and help corpus retrieval."""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

if "lancedb" not in sys.modules:
    sys.modules["lancedb"] = MagicMock()
if "pyarrow" not in sys.modules:
    sys.modules["pyarrow"] = MagicMock()

import numpy as np

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.composer_attachments import (  # noqa: E402
    ComposerAttachment,
    resolve_attachment_routing,
)
from core.help_corpus_manifest import HELP_DOC_SOURCE_PREFIX  # noqa: E402
from core.help_corpus_retrieval import (  # noqa: E402
    append_canonical_action_block,
    help_doc_ids_from_sources,
    log_help_query,
    match_canonical_answer,
)
from core.knowledge.registry import resolve_turn_knowledge_service  # noqa: E402
from core.knowledge.types import SERVICE_INTERNAL_CORPUS  # noqa: E402
from core.prompt_blocks import build_prompt_blocks  # noqa: E402
from mcp.rag_tool import (  # noqa: E402
    _filter_results_by_source_prefix,
    rag_search,
)


class TestHelpRouting(unittest.TestCase):
    def test_resolve_turn_knowledge_service_help(self) -> None:
        self.assertEqual(
            resolve_turn_knowledge_service(composer_tool="help"),
            SERVICE_INTERNAL_CORPUS,
        )

    def test_composer_help_routes_web(self) -> None:
        patch = resolve_attachment_routing(
            [ComposerAttachment(kind="tool", id="help", label="Help")]
        )
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch["route"], "web")
        self.assertEqual(patch["strategy"], "attachment_tool_help")
        self.assertEqual(patch["attachment_tool"], "help")

    def test_help_prompt_suffix_when_attached_with_sources(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            composer_help_attached=True,
            help_canonical_hint=" Canonical answer for this question: Settings → AI & Models.",
        )
        joined = " ".join(blocks.system_suffixes)
        self.assertIn("@[tool:help]", joined)
        self.assertIn("Settings → AI & Models", joined)

    def test_help_empty_suffix(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=False,
            help_sources_empty=True,
        )
        self.assertTrue(blocks.no_sources_mode)
        self.assertIn("Open Qube documentation", " ".join(blocks.system_suffixes))


class TestHelpCorpusRetrieval(unittest.TestCase):
    def test_match_canonical_answer_gpu_layers(self) -> None:
        entry = match_canonical_answer("Where are GPU layers in settings?")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry["doc_id"], "features.settings.ai_models")
        self.assertIn("AI & Models", entry["answer"])

    def test_match_canonical_answer_knowledge_settings(self) -> None:
        entry = match_canonical_answer("Where is Knowledge settings?")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry["doc_id"], "features.settings.knowledge")

    def test_match_canonical_ignores_generic_where_is(self) -> None:
        entry = match_canonical_answer("Where is something unrelated?")
        self.assertIsNone(entry)

    def test_help_doc_ids_from_sources(self) -> None:
        doc_ids = help_doc_ids_from_sources(
            [
                {
                    "filename": f"{HELP_DOC_SOURCE_PREFIX}features/settings/ai-models.md",
                    "content": "GPU layers",
                    "type": "web",
                },
                {"filename": "user-notes.pdf", "content": "ignore", "type": "web"},
            ]
        )
        self.assertEqual(doc_ids, ["features.settings.ai_models"])

    def test_log_help_query_emits_json(self) -> None:
        with self.assertLogs("Qube.Help", level="INFO") as logs:
            log_help_query(
                query="hide companion fullscreen",
                retrieved_doc_ids=["features.settings.companion_desktop"],
                canonical_id="features.settings.companion_desktop.fullscreen",
                session_id="sess-1",
            )
        payload = json.loads(logs.records[-1].message.split("[Help] ", 1)[1])
        self.assertEqual(payload["event"], "help_query")
        self.assertEqual(payload["retrieved_doc_ids"], ["features.settings.companion_desktop"])


    def test_append_canonical_action_block(self) -> None:
        entry = match_canonical_answer("Where are GPU layers in settings?")
        self.assertIsNotNone(entry)
        assert entry is not None
        out = append_canonical_action_block("Settings → AI & Models.", entry)
        self.assertIn("open_settings_section", out)
        self.assertIn("ai.models", out)


class TestRagSourcePrefixFilter(unittest.TestCase):
    def test_filter_results_by_source_prefix(self) -> None:
        rows = [
            {"source": "qube/documentation/a.md", "text": "help"},
            {"source": "main/report.pdf", "text": "user"},
        ]
        out = _filter_results_by_source_prefix(rows, HELP_DOC_SOURCE_PREFIX)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["source"].startswith(HELP_DOC_SOURCE_PREFIX))

    @patch("mcp.rag_tool.logger")
    def test_prefix_scoped_search(self, _log) -> None:
        store = MagicMock()
        table = MagicMock()
        store.table = table
        table.search.return_value.where.return_value.limit.return_value.to_list.return_value = [
            {
                "source": f"{HELP_DOC_SOURCE_PREFIX}faq/memory-vs-library.md",
                "text": "Memory stores facts; Library stores documents.",
                "chunk_id": 0,
            }
        ]

        result = rag_search(
            "memory vs library",
            np.zeros(768),
            store,
            top_k=3,
            source_prefix_filter=HELP_DOC_SOURCE_PREFIX,
        )
        self.assertIn(HELP_DOC_SOURCE_PREFIX, result["llm_context"])
        self.assertEqual(len(result["sources"]), 1)
        where_arg = table.search.return_value.where.call_args[0][0]
        self.assertIn("LIKE", where_arg)
        self.assertIn(HELP_DOC_SOURCE_PREFIX, where_arg)


if __name__ == "__main__":
    unittest.main()
