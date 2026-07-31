"""Tests for Pro Library depth features."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.capabilities import EditionTier, invalidate_capabilities_cache, resolve_capabilities
from core.chunking.precision_rerank import apply_precision_rerank_to_rag_hits
from core.chunking.semantic_ingest import semantic_breakpoint_chunks, split_sentences
from core.chunking.structure_chunker import ChunkRecord
from core.chunking.semantic_ingest import chunk_document_for_precision_ingest
from core.knowledge.document.types import Document, DocumentSection
from core import capabilities as capabilities_mod
from core import library_pro_features as pro_features


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for index, text in enumerate(texts):
            seed = float(len(text) + index)
            vectors.append([seed, seed / 2.0, seed / 3.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


class SemanticIngestTests(unittest.TestCase):
    def test_split_sentences(self) -> None:
        parts = split_sentences("First sentence. Second sentence!")
        self.assertEqual(len(parts), 2)

    def test_semantic_breakpoint_chunks(self) -> None:
        text = (
            "Revenue reached four point two million dollars. "
            "Operating margin improved to nineteen percent. "
            "Unrelated topic about migration birds."
        )
        chunks = semantic_breakpoint_chunks(text, _FakeEmbedder(), max_chars=500)
        self.assertGreaterEqual(len(chunks), 1)

    def test_precision_ingest_expands_large_section(self) -> None:
        long_body = (
            "Alpha sentence one about revenue. Alpha sentence two about margin. "
            "Beta sentence begins a different topic entirely here."
        ) * 16
        document = Document(
            url="file://sample.md",
            title="Doc",
            sections=[
                DocumentSection(
                    heading="Section",
                    level=2,
                    text=long_body,
                )
            ],
        )
        records = chunk_document_for_precision_ingest(document, _FakeEmbedder())
        self.assertGreater(len(records), 1)


class PrecisionRerankTests(unittest.TestCase):
    def test_rerank_orders_by_query_similarity(self) -> None:
        query_vector = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        hits = [
            {"text": "weak", "vector": [0.1, 0.9, 0.0, 0.0]},
            {"text": "strong", "vector": [0.95, 0.05, 0.0, 0.0]},
        ]
        ranked = apply_precision_rerank_to_rag_hits(
            query_vector,
            hits,
            embedder=_FakeEmbedder(),
        )
        self.assertEqual(ranked[0]["text"], "strong")


class ProFeatureGatingTests(unittest.TestCase):
    def tearDown(self) -> None:
        invalidate_capabilities_cache()

    def test_home_tier_denies_pro_library_features(self) -> None:
        caps = resolve_capabilities(tier=EditionTier.HOME, source="test")
        self.assertFalse(caps.has(pro_features.PRO_INGEST_CAPABILITY))
        self.assertFalse(caps.has(pro_features.PRO_RERANK_CAPABILITY))

    def test_pro_tier_grants_library_features(self) -> None:
        caps = resolve_capabilities(tier=EditionTier.PRO, source="test")
        self.assertTrue(caps.has(pro_features.PRO_INGEST_CAPABILITY))
        self.assertTrue(caps.has(pro_features.PRO_RERANK_CAPABILITY))

    @patch("core.app_settings.get_library_precision_ingest_enabled", return_value=True)
    @patch("core.library_pro_features.user_has_pro_library_ingest", return_value=False)
    def test_precision_ingest_requires_license(self, *_mocks) -> None:
        self.assertFalse(pro_features.precision_ingest_enabled())

    @patch("core.app_settings.get_library_precision_rerank_enabled", return_value=True)
    @patch("core.library_pro_features.user_has_pro_library_rerank", return_value=True)
    def test_precision_rerank_enabled_with_license(self, *_mocks) -> None:
        self.assertTrue(pro_features.precision_rerank_enabled())

    def test_grant_all_override_still_grants_all_when_enabled(self) -> None:
        original = capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE
        capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE = True
        invalidate_capabilities_cache()
        try:
            caps = resolve_capabilities()
            self.assertTrue(caps.has(pro_features.PRO_INGEST_CAPABILITY))
        finally:
            capabilities_mod._GRANT_ALL_CAPABILITIES_OVERRIDE = original
            invalidate_capabilities_cache()


if __name__ == "__main__":
    unittest.main()
