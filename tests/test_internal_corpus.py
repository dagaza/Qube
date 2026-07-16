"""Tests for internal corpus knowledge service (Phase 6 Slice 2)."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.composer_attachments import (  # noqa: E402
    ComposerAttachment,
    resolve_attachment_routing,
)
from core.knowledge.registry import resolve_turn_knowledge_service  # noqa: E402
from core.knowledge.types import (  # noqa: E402
    RetrievalContext,
    SERVICE_INTERNAL_CORPUS,
)
from core.knowledge.web_retrieval import run_v2_web_retrieval  # noqa: E402


_LIBRARY_RAG_SOURCES = [
    {
        "id": 1,
        "filename": "eval_kubernetes_notes.md",
        "content": (
            "On our staging cluster, ingress is handled by NGINX Ingress Controller v1.11. "
            "TLS terminates at the ingress using cert-manager."
        ),
        "type": "rag",
        "chunk_id": "eval_kubernetes_notes.md::0",
        "semantic_score": 0.82,
    }
]


class TestInternalCorpusKnowledge(unittest.TestCase):
    def test_resolve_turn_knowledge_service_library(self) -> None:
        self.assertEqual(
            resolve_turn_knowledge_service(composer_tool="library"),
            SERVICE_INTERNAL_CORPUS,
        )

    def test_composer_library_routes_web(self) -> None:
        patch = resolve_attachment_routing(
            [ComposerAttachment(kind="tool", id="library", label="Library")]
        )
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch["route"], "web")
        self.assertEqual(patch["strategy"], "attachment_tool_library")
        self.assertEqual(patch["attachment_tool"], "library")

    @patch("core.knowledge.pipeline_internal_corpus.search_library_chunks")
    def test_v2_internal_corpus_bundle(self, mock_search) -> None:
        mock_search.return_value = (
            [
                {
                    "title": "eval_kubernetes_notes.md",
                    "snippet": _LIBRARY_RAG_SOURCES[0]["content"],
                    "source": "eval_kubernetes_notes.md",
                    "chunk_id": "eval_kubernetes_notes.md::0",
                    "full_text": _LIBRARY_RAG_SOURCES[0]["content"],
                    "_library_semantic_score": 0.82,
                }
            ],
            [],
        )
        store = MagicMock()

        outcome = run_v2_web_retrieval(
            query="How is ingress configured on staging?",
            semantic_query="How is ingress configured on staging?",
            knowledge_service=SERVICE_INTERNAL_CORPUS,
            query_vector=np.zeros(384, dtype=np.float32),
            library_store=store,
        )

        self.assertIsNotNone(outcome.bundle)
        assert outcome.bundle is not None
        self.assertEqual(outcome.bundle.knowledge_service, SERVICE_INTERNAL_CORPUS)
        self.assertEqual(len(outcome.bundle.sources), 1)
        source = outcome.bundle.sources[0]
        self.assertEqual(source.adapter, "lancedb_library")
        self.assertEqual(source.document_type, "library_chunk")
        self.assertIn(source.fetch_status, {"snippet", "full_text"})
        self.assertEqual(outcome.bundle.adapter_calls, ("lancedb_library",))
        mock_search.assert_called_once()

    def test_empty_bundle_without_store(self) -> None:
        from core.knowledge.pipeline_internal_corpus import InternalCorpusEvidencePipeline

        pipeline = InternalCorpusEvidencePipeline()
        bundle, rel_diag, raw = pipeline.run(
            RetrievalContext(
                query="test",
                semantic_query="test",
                knowledge_service=SERVICE_INTERNAL_CORPUS,
                query_vector=np.zeros(384, dtype=np.float32),
            )
        )
        self.assertEqual(bundle.knowledge_service, SERVICE_INTERNAL_CORPUS)
        self.assertEqual(bundle.sources, ())
        self.assertEqual(bundle.stop_reason, "no_library_store")
        self.assertIsNone(rel_diag)
        self.assertEqual(raw, [])


if __name__ == "__main__":
    unittest.main()
