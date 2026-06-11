"""Poisoned-history protection for degenerated assistant completions."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.history_degeneration import (  # noqa: E402
    HISTORY_SUPPRESSION_PLACEHOLDER,
    resolve_assistant_history_content,
    score_history_degeneration,
)

_EXCHANGE_15_PARTIAL = (
    "Here is a comprehensive analysis of Nepal's music and arts scene.\n\n"
    "#### Artistic Medium Breakdown (List Format)\n\n"
    "*   **"
)

_LOG_DEGENERATE_TAIL = (
    "Birds take to‑bath in a few different ways, and the reasons are pretty practical.  \n"
    "1. **Cleaning** – The most obvious reason is to keep their feathers clean and free "
    "from dirt, parasites, and oil‑rich prey.  \n\n"
    "2. **Pre‑‑treatment –** The **‑ 3 – – ??  \n"
    "The …‑…  \n"
    "**We … ...**  \n"
    "We … ­  \n"
    "We………  \n"
)

_VALID_TWO_POINT = (
    "Birds bathe for hygiene and comfort.\n\n"
    "1. **Cleaning** – Removes dirt and parasites from feathers.\n\n"
    "2. **Temperature** – Splashing helps birds cool down on hot days.\n"
)


class TestHistoryDegeneration(unittest.TestCase):
    def test_clean_response_not_suppressed(self) -> None:
        result = score_history_degeneration(_VALID_TWO_POINT)
        self.assertFalse(result.should_suppress)
        stored, resolved = resolve_assistant_history_content(_VALID_TWO_POINT)
        self.assertEqual(stored, _VALID_TWO_POINT.strip())
        self.assertFalse(resolved.should_suppress)

    def test_degenerate_tail_suppressed(self) -> None:
        result = score_history_degeneration(_LOG_DEGENERATE_TAIL)
        self.assertTrue(result.should_suppress)
        self.assertIn("harmony_degeneration", result.flags)
        stored, _ = resolve_assistant_history_content(_LOG_DEGENERATE_TAIL)
        self.assertEqual(stored, HISTORY_SUPPRESSION_PLACEHOLDER)

    def test_pure_number_topic_not_applicable_here_but_orphan_numbering(self) -> None:
        text = "Kathmandu attractions include:\n\n1.\n"
        result = score_history_degeneration(text)
        self.assertTrue(result.should_suppress)
        self.assertIn("unfinished_numbering", result.flags)

    def test_abrupt_ellipsis_cutoff(self) -> None:
        text = (
            "Kathmandu is home to religious minorities, including Muslims and Christians, "
            "and followers of indigenous traditions such as the Newar people's worship of local deities. "
            "In addition, small pockets of Kirghiz add diversity, giving a vibrant, inclusive‑tied…"
        )
        result = score_history_degeneration(text)
        self.assertTrue(result.should_suppress)
        self.assertIn("abrupt_cutoff", result.flags)

    def test_meta_commentary_alone_below_threshold(self) -> None:
        text = "Sorry, I don't have that information."
        result = score_history_degeneration(text)
        self.assertFalse(result.should_suppress)

    def test_trace_fields(self) -> None:
        result = score_history_degeneration(_LOG_DEGENERATE_TAIL)
        fields = result.trace_fields()
        self.assertTrue(fields["history_degeneration_suspect"])
        self.assertTrue(fields["history_degeneration_suppressed"])
        self.assertGreater(float(fields["history_degeneration_score"]), 0.5)

    def test_exchange_15_partial_not_suppressed_after_cancel(self) -> None:
        result = score_history_degeneration(_EXCHANGE_15_PARTIAL, stream_cancelled=True)
        self.assertFalse(result.should_suppress)
        stored, resolved = resolve_assistant_history_content(
            _EXCHANGE_15_PARTIAL,
            stream_cancelled=True,
        )
        self.assertEqual(stored, _EXCHANGE_15_PARTIAL.strip())
        self.assertFalse(resolved.should_suppress)

    def test_degenerate_tail_still_suppressed_after_cancel(self) -> None:
        result = score_history_degeneration(_LOG_DEGENERATE_TAIL, stream_cancelled=True)
        self.assertTrue(result.should_suppress)
        stored, _ = resolve_assistant_history_content(
            _LOG_DEGENERATE_TAIL,
            stream_cancelled=True,
        )
        self.assertEqual(stored, HISTORY_SUPPRESSION_PLACEHOLDER)


if __name__ == "__main__":
    unittest.main()
