"""Request-frame stripping for retrieval queries (plan S5)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.knowledge.discovery.types import CandidateUrl, DiscoveryResult  # noqa: E402
from core.knowledge.pipeline_general_web import run_general_web_evidence_pipeline  # noqa: E402
from core.knowledge.pipeline_trusted import TrustedEvidencePipeline  # noqa: E402
from core.knowledge.search_outcome import SearchOutcome, SearchOutcomeKind  # noqa: E402
from core.knowledge.types import (  # noqa: E402
    SERVICE_TRUSTED_KNOWLEDGE,
    RetrievalBudget,
    RetrievalContext,
)
from core.retrieval_query_cleanup import strip_request_frames  # noqa: E402


class StripRequestFramesTests(unittest.TestCase):
    def test_strips_frames(self) -> None:
        cases = [
            (
                "Yes, please tell me how ANT analyzes Artificial Inteligence for example.",
                "how ANT analyzes Artificial Inteligence",
            ),
            ("Yes, tell me how MS is treated", "how MS is treated"),
            ("Sure! Can you explain how statins work?", "how statins work?"),
            ("ok, could you please tell me about heat pumps, thanks", "heat pumps"),
            ("Please, what is OAuth?", "what is OAuth?"),
            ("please tell me more about the JIT compiler", "the JIT compiler"),
            ("What is actor-network theory?>", "What is actor-network theory?"),
            ("What is actor-network theory>", "What is actor-network theory"),
            ("give an overview of ANT, for instance.", "give an overview of ANT"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(strip_request_frames(text), expected)

    def test_leaves_questions_alone(self) -> None:
        # Boundary-only: nothing inside the question is removed.
        unchanged = [
            "example-based learning for robots",
            "for example, how do vaccines work",  # leading "for example" is content-bearing
            "Tell Me Lies song meaning",
            "Please Please Me album",
            "Yes Minister episodes",
            "Thanks for the memories lyrics",
            "OK Computer review",
            "what is x > 5 in python",
            "What does the literature say about dapagliflozin in HFrEF?",
            "Explain U.S. politics.",
        ]
        for text in unchanged:
            with self.subTest(text=text):
                self.assertEqual(strip_request_frames(text), text)

    def test_frame_only_text_is_returned_unchanged(self) -> None:
        # Nothing to search for once the frame is gone: keep the original
        # (S6 resolves bare affirmations from the conversation).
        for text in ("Yes, please.", "Please.", "OK, thanks.", "sure!", "", "   "):
            with self.subTest(text=text):
                self.assertEqual(strip_request_frames(text), text.strip())


def _discovery(*urls: str) -> DiscoveryResult:
    candidates = tuple(
        CandidateUrl(url=u, title="Multiple sclerosis treatment", snippet="MS therapies.", source="duckduckgo")
        for u in urls
    )
    return DiscoveryResult(
        candidates=candidates,
        raw_rows=tuple({"title": c.title, "snippet": c.snippet, "url": c.url} for c in candidates),
        search_outcome=SearchOutcome(kind=SearchOutcomeKind.SERP_SUCCESS, candidate_count=len(candidates)),
    )


class WebPathUsesCleanQueryTests(unittest.TestCase):
    RAW = "Yes, tell me how multiple sclerosis (MS) is treated"
    CLEAN = "how multiple sclerosis (MS) is treated"

    @patch("core.knowledge.pipeline_general_web.discover_full")
    def test_general_web_discovers_with_clean_query(self, mock_discover) -> None:
        mock_discover.return_value = _discovery()
        ctx = RetrievalContext(query=self.RAW, semantic_query=self.RAW, budget=RetrievalBudget(max_results=3))
        bundle, _diag, _raw = run_general_web_evidence_pipeline(ctx)
        self.assertEqual(mock_discover.call_args.args[0], self.CLEAN)
        self.assertEqual(bundle.query_raw, self.RAW)
        self.assertEqual(bundle.query_resolved, self.CLEAN)

    @patch("core.knowledge.pipeline_trusted.discover_full_with_fallback")
    @patch("core.knowledge.pipeline_trusted.search_wikipedia")
    def test_trusted_searches_with_clean_query(self, mock_wiki, mock_discover) -> None:
        mock_wiki.return_value = []
        mock_discover.return_value = _discovery()
        ctx = RetrievalContext(
            query=self.RAW,
            semantic_query=self.RAW,
            knowledge_service=SERVICE_TRUSTED_KNOWLEDGE,
            budget=RetrievalBudget(max_results=3),
        )
        TrustedEvidencePipeline().run(ctx)
        self.assertEqual(mock_wiki.call_args.args[0], self.CLEAN)
        self.assertEqual(mock_discover.call_args.args[0], self.CLEAN)

    @patch("core.knowledge.pipeline_general_web.discover_full")
    def test_plain_question_is_unchanged(self, mock_discover) -> None:
        mock_discover.return_value = _discovery()
        q = "What is the boiling point of ethanol?"
        run_general_web_evidence_pipeline(
            RetrievalContext(query=q, semantic_query=q, budget=RetrievalBudget(max_results=3))
        )
        self.assertEqual(mock_discover.call_args.args[0], q)


if __name__ == "__main__":
    unittest.main()
