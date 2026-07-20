"""Tests for canonical action wiring and production help retrieval eval."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

if "lancedb" not in sys.modules:
    sys.modules["lancedb"] = MagicMock()
if "pyarrow" not in sys.modules:
    sys.modules["pyarrow"] = MagicMock()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.help_corpus_retrieval import (  # noqa: E402
    append_canonical_action_block,
    lookup_manifest_action,
    match_canonical_answer,
)
from core.help_production_eval import (  # noqa: E402
    evaluate_production_help_retrieval,
    rank_help_docs_via_rag,
)


class HelpCanonicalActionTests(unittest.TestCase):
    def test_lookup_gpu_layers_action(self) -> None:
        action = lookup_manifest_action("open_settings_ai_models")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action["kind"], "open_settings_section")
        self.assertEqual(action["settings_section"], "ai.models")

    def test_append_canonical_action_block(self) -> None:
        entry = match_canonical_answer("Where are GPU layers in settings?")
        self.assertIsNotNone(entry)
        assert entry is not None
        out = append_canonical_action_block("Open Settings → AI & Models.", entry)
        self.assertIn("[action:open_settings_section", out)
        self.assertIn("ai.models", out)


class HelpProductionEvalTests(unittest.TestCase):
    def test_gpu_layers_ranks_ai_models(self) -> None:
        ranked, rag_pool = rank_help_docs_via_rag(
            "Where are GPU layers in settings?", top_k=3
        )
        self.assertTrue(ranked)
        self.assertIn("features.settings.ai_models", ranked[:3])
        self.assertIn("features.settings.ai_models", rag_pool)

    def test_production_eval_runs(self) -> None:
        summary = evaluate_production_help_retrieval()
        positive = summary.total - summary.negative_total
        self.assertGreater(positive, 30)
        self.assertGreaterEqual(summary.top1_rate, 0.75)


if __name__ == "__main__":
    unittest.main()
