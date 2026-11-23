"""Phase 0/1 knowledge-quality fixes: history strip, prompt contract, evidence pack, scores."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.bundle_builder import (  # noqa: E402
    _compute_scientific_confidence,
    _compute_scientific_coverage,
    _scientific_row_to_evidence,
    build_scientific_evidence_bundle,
)
from core.knowledge.quality_eval import evaluate_ant_fixture, load_json  # noqa: E402
from core.knowledge.ranking.reliability import apply_reliability_scores  # noqa: E402
from core.knowledge.types import COVERAGE_POOR, EvidenceObject  # noqa: E402
from core.knowledge.ui_adapter import (  # noqa: E402
    evidence_body_for_prompt,
    evidence_to_ui_source,
)
from core.memory_filters import (  # noqa: E402
    CITATION_DISCIPLINE_SUFFIX,
    SCIENTIFIC_MEDICAL_DISCLAIMER_SUFFIX,
)
from core.prompt_blocks import build_prompt_blocks, compose_system_prompt  # noqa: E402
from core.prompt_history import strip_composer_tokens_from_history  # noqa: E402
from core.prompt_renderers import render_system_ok_messages  # noqa: E402


class TestComposerHistoryStrip(unittest.TestCase):
    def test_strips_tool_tokens_from_user_turns(self) -> None:
        history = [
            {"role": "user", "content": "@[tool:evidence] What is OAuth?"},
            {"role": "assistant", "content": "OAuth is…"},
            {
                "role": "user",
                "content": "@[tool:evidence] Yes, please tell me how ANT analyzes AI.",
            },
        ]
        cleaned = strip_composer_tokens_from_history(history)
        self.assertEqual(cleaned[0]["content"], "What is OAuth?")
        self.assertIn("ANT analyzes AI", cleaned[2]["content"])
        self.assertNotIn("@[tool:", cleaned[2]["content"])
        self.assertEqual(cleaned[1]["content"], "OAuth is…")


class TestPromptContract(unittest.TestCase):
    def test_medical_disclaimer_keeps_scientific_persona_and_citations(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_source_count=2,
            web_hit_count=2,
            scientific_medical_disclaimer=True,
            knowledge_service="scientific_evidence",
            current_date="2026-11-23",
            engine_mode="external",
        )
        sys_p = compose_system_prompt(blocks)
        self.assertIn("Scientific literature results", sys_p)
        self.assertIn(CITATION_DISCIPLINE_SUFFIX.strip()[:40], sys_p)
        self.assertIn(SCIENTIFIC_MEDICAL_DISCLAIMER_SUFFIX.strip()[:30], sys_p)
        self.assertIn("Today's date is 2026-11-23", sys_p)

    def test_citation_exemplar_allows_abstention(self) -> None:
        blocks = build_prompt_blocks(
            execution_route="WEB",
            explicit_remember_active=False,
            has_retrieval_sources=True,
            retrieval_context="--- [1]: Paper ---\nAbstract text.",
            retrieval_source_count=1,
            web_hit_count=1,
            knowledge_service="scientific_evidence",
            conversation_history=[{"role": "user", "content": "Explain ANT and AI."}],
            current_date="",
            engine_mode="external",
        )
        last = render_system_ok_messages(blocks)[-1]["content"]
        self.assertIn("Cite [1] only on sentences that source directly supports.", last)
        self.assertIn("If the source is not relevant, say so plainly without brackets.", last)


class TestEvidencePack(unittest.TestCase):
    def test_prefers_full_text_and_metadata(self) -> None:
        obj = EvidenceObject(
            id="ek_1",
            source_id="doi:1",
            adapter="openalex",
            retrieval_method="abstract",
            title="Actor-Network Theory and AI",
            excerpt="Short excerpt only.",
            full_text="Full abstract " * 80,
            url="https://example.org/ant-ai",
            document_type="journal_abstract",
            venue="STS Journal",
            publication_date="2022",
            authors=("Latour, B", "Example, A"),
            peer_reviewed=True,
            preprint=False,
            relevance_score=0.7,
            authority_score=0.8,
            reliability_score=0.7,
            freshness_score=0.85,
            fetch_status="abstract",
        )
        body = evidence_body_for_prompt(obj, char_budget=200)
        self.assertTrue(body.startswith("Full abstract"))
        self.assertLessEqual(len(body), 200)
        row = evidence_to_ui_source(obj, ui_id=1)
        self.assertEqual(row["venue"], "STS Journal")
        self.assertEqual(row["publication_date"], "2022")
        self.assertTrue(row["peer_reviewed"])
        self.assertIn("Full abstract", row["content"])


class TestScoreBugs(unittest.TestCase):
    def test_reliability_and_freshness_persist_on_scientific_row(self) -> None:
        row = {
            "title": "Semaglutide outcomes",
            "snippet": "Semaglutide was effective and safe.",
            "full_text": "Semaglutide was effective and safe in the primary analysis.",
            "_adapter": "pubmed",
            "publication_date": "2024-01",
            "_scientific_relevance": 0.8,
            "_reliability_score": 0.66,
            "peer_reviewed": True,
        }
        obj = _scientific_row_to_evidence(row, index=1, retrieved_at=0.0)
        self.assertAlmostEqual(obj.reliability_score, 0.66, places=2)
        self.assertIsNotNone(obj.freshness_score)
        self.assertGreaterEqual(float(obj.freshness_score or 0.0), 0.85)

    def test_reliability_skips_self_by_doi(self) -> None:
        rows = [
            {
                "title": "A",
                "snippet": "effective and beneficial",
                "doi": "10.1/a",
                "_scientific_relevance": 0.5,
            },
            {
                "title": "B",
                "snippet": "no benefit and adverse",
                "doi": "10.1/b",
                "_scientific_relevance": 0.5,
            },
        ]
        scored = apply_reliability_scores(rows)
        self.assertEqual(len(scored), 2)
        # Identity skip must not inflate scores via self-similarity.
        for row in scored:
            self.assertIn("_reliability_score", row)
            self.assertLessEqual(row["_reliability_score"], 0.95)

    def test_low_token_relevance_caps_coverage_poor(self) -> None:
        sources = (
            EvidenceObject(
                id="ek_1",
                source_id="a",
                adapter="arxiv",
                retrieval_method="abstract",
                title="Unrelated",
                excerpt="noise",
                full_text="noise",
                url=None,
                document_type="preprint",
                relevance_score=0.1,
                authority_score=0.7,
                fetch_status="abstract",
            ),
            EvidenceObject(
                id="ek_2",
                source_id="b",
                adapter="openalex",
                retrieval_method="abstract",
                title="Also unrelated",
                excerpt="noise",
                full_text="noise",
                url=None,
                document_type="journal_abstract",
                relevance_score=0.11,
                authority_score=0.7,
                fetch_status="abstract",
            ),
        )
        coverage, rationale = _compute_scientific_coverage(sources)
        self.assertEqual(coverage, COVERAGE_POOR)
        self.assertIn("low", rationale.lower())
        conf = _compute_scientific_confidence(sources)
        high = _compute_scientific_confidence(
            tuple(
                EvidenceObject(
                    id=s.id,
                    source_id=s.source_id,
                    adapter=s.adapter,
                    retrieval_method=s.retrieval_method,
                    title=s.title,
                    excerpt=s.excerpt,
                    full_text=s.full_text,
                    url=s.url,
                    document_type=s.document_type,
                    relevance_score=0.8,
                    authority_score=s.authority_score,
                    fetch_status=s.fetch_status,
                )
                for s in sources
            )
        )
        self.assertLess(conf, high)

    @patch("core.knowledge.adapters.openalex.knowledge_get")
    def test_openalex_does_not_mark_preprint_peer_reviewed(self, mock_get) -> None:
        from core.knowledge.adapters import openalex as oa

        class _Resp:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "results": [
                        {
                            "display_name": "A preprint on ants",
                            "type": "preprint",
                            "abstract_inverted_index": {"Hello": [0], "world": [1]},
                            "primary_location": {
                                "source": {
                                    "display_name": "arXiv",
                                    "type": "repository",
                                }
                            },
                            "authorships": [],
                            "publication_year": 2020,
                            "open_access": {"is_oa": True},
                        }
                    ]
                }

        mock_get.return_value = _Resp()
        rows = oa.search_openalex("ants", max_results=1)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["preprint"])
        self.assertFalse(rows[0]["peer_reviewed"])


class TestPhase0FixtureBaseline(unittest.TestCase):
    def test_ant_fixture_phase1_contract(self) -> None:
        fixture = load_json(
            "eval/fixtures/knowledge_quality/ant_ai_candidates.json"
        )
        report = evaluate_ant_fixture(fixture)
        self.assertTrue(report["composer_token_stripped"])
        self.assertTrue(report["prompt_allows_abstention"])
        self.assertTrue(report["scientific_persona"])
        self.assertTrue(report["date_injected"])
        self.assertTrue(report["evidence_pack_has_metadata"])
        self.assertTrue(report["freshness_populated"])
        # Documented Phase 0 gap: token ranker still keeps reject titles.
        self.assertTrue(report["phase0_known_gap_token_ranker_keeps_irrelevant"])


if __name__ == "__main__":
    unittest.main()
