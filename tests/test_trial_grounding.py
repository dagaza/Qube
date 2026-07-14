"""Tests for trial grounding ranking patch."""

from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.knowledge.ranking.relevance import score_rows  # noqa: E402
from core.knowledge.ranking.trial_grounding import (  # noqa: E402
    extract_trial_signal,
    trial_grounding_boost,
)

_QA_3B = (
    "Summarize key outcomes from the EMPEROR-Reduced trial for heart failure."
)
_QA_3C = "What does the literature say about dapagliflozin in HFrEF?"


class TestTrialGrounding(unittest.TestCase):
    def test_extract_trial_signal_emperor(self) -> None:
        signals = extract_trial_signal(_QA_3B)
        self.assertIn("emperor-reduced", signals)

    def test_extract_trial_signal_none_for_dapagliflozin_only(self) -> None:
        signals = extract_trial_signal(_QA_3C)
        self.assertEqual(signals, frozenset())

    def test_boost_prefers_trial_title_and_rct(self) -> None:
        signals = extract_trial_signal(_QA_3B)
        secondary = {
            "title": "Gliflozins in Practice: Real-Life Use of Dapagliflozin and Empagliflozin in HFrEF",
            "snippet": "Real-world SGLT2 inhibitor use in heart failure.",
            "full_text": "Real-world SGLT2 inhibitor use in heart failure.",
            "_adapter": "pubmed",
            "publication_types": ("Journal Article",),
        }
        trial_sub = {
            "title": "Empagliflozin in Heart Failure and Previous Coronary Revascularization: Insights From EMPEROR-Pooled",
            "snippet": "Analysis from the EMPEROR-Reduced and EMPEROR-Preserved trials.",
            "full_text": "Analysis from the EMPEROR-Reduced and EMPEROR-Preserved trials.",
            "_adapter": "pubmed",
            "publication_types": ("Journal Article", "Randomized Controlled Trial"),
        }
        self.assertGreater(
            trial_grounding_boost(trial_sub, signals),
            trial_grounding_boost(secondary, signals),
        )

    def test_qa_3b_ranking_reorders_pubmed_candidates(self) -> None:
        signals = extract_trial_signal(_QA_3B)
        rows = [
            {
                "title": "Gliflozins in Practice: Real-Life Use of Dapagliflozin and Empagliflozin in HFrEF",
                "snippet": "Heart failure real-world evidence.",
                "full_text": "Heart failure real-world evidence.",
                "_adapter": "pubmed",
                "publication_types": ("Journal Article",),
            },
            {
                "title": "Polycythemia, SGLT2 inhibitors, and associated outcomes across the cardio-kidney-metabolic spectrum",
                "snippet": "Pooled analysis including EMPEROR-Reduced.",
                "full_text": "Pooled analysis including EMPEROR-Reduced.",
                "_adapter": "pubmed",
                "publication_types": ("Journal Article", "Review"),
            },
            {
                "title": "Heart failure outcomes and empagliflozin effects in patients with heart failure and reduced ejection fraction",
                "snippet": "Post hoc analysis from EMPEROR-Reduced.",
                "full_text": "Post hoc analysis from EMPEROR-Reduced.",
                "_adapter": "pubmed",
                "publication_types": ("Journal Article", "Randomized Controlled Trial", "Multicenter Study"),
            },
        ]
        kept, _ = score_rows(
            rows,
            query=_QA_3B,
            trial_signals=signals,
        )
        self.assertIn("Heart failure outcomes and empagliflozin", kept[0]["title"])
        self.assertNotIn("Gliflozins in Practice", kept[0]["title"])

    def test_qa_3c_unchanged_without_trial_signal(self) -> None:
        rows = [
            {
                "title": "Qualitative evaluation to understand barriers and facilitators to prescribing SGLT2 inhibitors",
                "snippet": "Heart failure with reduced ejection fraction prescribing barriers.",
                "full_text": "Heart failure with reduced ejection fraction prescribing barriers.",
                "_adapter": "openalex",
            },
            {
                "title": "Comparative effectiveness of dapagliflozin vs. empagliflozin on major adverse cardiovascular events",
                "snippet": "Dapagliflozin in HFrEF patients.",
                "full_text": "Dapagliflozin in HFrEF patients.",
                "_adapter": "pubmed",
                "publication_types": ("Journal Article",),
            },
        ]
        kept_plain, _ = score_rows(rows, query=_QA_3C, trial_signals=None)
        kept_trial, _ = score_rows(
            rows, query=_QA_3C, trial_signals=extract_trial_signal(_QA_3C)
        )
        self.assertEqual(
            [r["title"] for r in kept_plain],
            [r["title"] for r in kept_trial],
        )


if __name__ == "__main__":
    unittest.main()
