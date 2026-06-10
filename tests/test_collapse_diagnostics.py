"""Collapse diagnostics for model degradation onset."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.collapse_diagnostics import (  # noqa: E402
    build_collapse_timeline,
    compute_collapse_diagnostics,
    score_format_drift,
    score_hallucination_indicators,
)
from core.history_degeneration import HISTORY_SUPPRESSION_PLACEHOLDER  # noqa: E402

_DEGENERATE = (
    "Attractions include:\n\n1. **Temple** – A landmark.\n\n"
    "2. **Pre‑‑treatment –** The **‑ 3 – – ??  \nWe………  \n"
)


class TestCollapseDiagnostics(unittest.TestCase):
    def test_clean_turn_low_risk(self) -> None:
        diag = compute_collapse_diagnostics(
            prompt="System\nUser: hello",
            output="Kathmandu is the capital of Nepal.",
            user_query="What is the capital of Nepal?",
            turn_index=0,
        )
        self.assertEqual(diag.collapse_risk, "LOW")
        self.assertLess(diag.collapse_score, 0.38)

    def test_degenerate_output_high_risk(self) -> None:
        diag = compute_collapse_diagnostics(
            prompt="x" * 9000,
            output=_DEGENERATE,
            user_query="What are the main attractions?",
            turn_index=7,
        )
        self.assertIn(diag.collapse_risk, ("MEDIUM", "HIGH"))
        self.assertGreaterEqual(diag.degeneration_score, 0.5)
        fields = diag.trace_fields()
        self.assertEqual(fields["collapse_risk"], diag.collapse_risk)
        self.assertIn("collapse_prompt_length", fields)

    def test_orphan_web_citation_hallucination(self) -> None:
        score, flags = score_hallucination_indicators(
            user_query="weather today",
            output="It is sunny [W]",
        )
        self.assertIn("orphan_web_citation", flags)
        self.assertGreater(score, 0.0)

    def test_template_leak_format_drift(self) -> None:
        score, flags = score_format_drift("Answer <|channel|>final text")
        self.assertGreater(score, 0.0)
        self.assertTrue(flags)

    def test_build_timeline_from_dict_turns(self) -> None:
        timeline = build_collapse_timeline(
            [
                {
                    "turn_index": 0,
                    "user_message": "hello",
                    "trace": {
                        "prompt": "p",
                        "output": "fine answer",
                        "metadata": {},
                    },
                },
                {
                    "turn_index": 1,
                    "user_message": "follow up",
                    "trace": {
                        "prompt": "p" * 200,
                        "output": HISTORY_SUPPRESSION_PLACEHOLDER,
                        "metadata": {"history_degeneration_suppressed": True},
                    },
                },
            ],
            backend_label="qube",
        )
        self.assertEqual(len(timeline), 2)
        self.assertIn("collapse_risk", timeline[0])
        self.assertEqual(timeline[0]["backend"], "qube")

    def test_unesco_denial_after_suppressed_prior_is_high_risk(self) -> None:
        turn8 = (
            "There are no UNESCO World Heritage Sites located within the city "
            "limits of Kathmandu itself."
        )
        diag = compute_collapse_diagnostics(
            prompt="p" * 1000,
            output=turn8,
            user_query="Which UNESCO World Heritage sites are in the city?",
            turn_index=8,
            prior_turn_suppressed=True,
        )
        self.assertIn("denies_established_fact", diag.hallucination_flags)
        self.assertIn("prior_turn_suppressed_confabulation", diag.hallucination_flags)
        self.assertEqual(diag.collapse_risk, "HIGH")


if __name__ == "__main__":
    unittest.main()
