"""Unified output degeneration detector."""
from __future__ import annotations

import os
import sys
import unittest

_WS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

from core.output_degeneration import (  # noqa: E402
    OutputDegenerationStreamObserver,
    detect_output_degeneration,
    should_mark_turn_unreliable,
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


class TestOutputDegenerationDetector(unittest.TestCase):
    def test_clean_response_low_risk(self) -> None:
        result = detect_output_degeneration(_VALID_TWO_POINT)
        self.assertEqual(result.risk, "LOW")
        self.assertFalse(should_mark_turn_unreliable(result))

    def test_degenerate_log_tail_high_risk(self) -> None:
        result = detect_output_degeneration(_LOG_DEGENERATE_TAIL)
        self.assertEqual(result.risk, "HIGH")
        self.assertTrue(should_mark_turn_unreliable(result))

    def test_composite_weights_applied(self) -> None:
        result = detect_output_degeneration(_LOG_DEGENERATE_TAIL)
        c = result.components
        expected = min(
            1.0,
            max(c.repetition, c.entropy_collapse * 0.85) * 0.35
            + max(c.malformed_list, c.unfinished_bullet, c.markdown_explosion * 0.75) * 0.20
            + max(c.meta_commentary, c.self_correction) * 0.20
            + c.punctuation_loop * 0.15
            + c.truncation * 0.10,
        )
        self.assertAlmostEqual(result.composite_score, expected, places=2)

    def test_orphan_numbered_list(self) -> None:
        text = "Kathmandu attractions include:\n\n1.\n"
        result = detect_output_degeneration(text)
        self.assertTrue(should_mark_turn_unreliable(result))
        self.assertIn("unfinished_bullet", result.flags)

    def test_repetition_token_loop(self) -> None:
        text = " ".join(["[W]"] * 12)
        result = detect_output_degeneration(text)
        self.assertEqual(result.risk, "HIGH")
        self.assertIn("repetition", result.flags)

    def test_meta_apology_alone_stays_low(self) -> None:
        result = detect_output_degeneration("Sorry, I don't have that information.")
        self.assertFalse(should_mark_turn_unreliable(result))

    def test_factual_population_prose_not_cut_off(self) -> None:
        text = (
            "The city of Kathmandu has a population that is around 1 million people. "
            "In fact, the latest estimates put it at roughly 1,442,271 residents for 2026, "
            "making it Nepal's largest metropolis and the economic hub of the country."
        )
        result = detect_output_degeneration(text)
        self.assertEqual(result.risk, "LOW")
        observer = OutputDegenerationStreamObserver(rescore_every=80)
        tripped = any(
            observer.observe(text[i : i + 20])
            for i in range(0, len(text), 20)
        )
        self.assertFalse(tripped)

    def test_stream_observer_trips_on_degenerate_tail(self) -> None:
        observer = OutputDegenerationStreamObserver(rescore_every=80)
        tripped = False
        chunk = _LOG_DEGENERATE_TAIL
        for i in range(0, len(chunk), 20):
            if observer.observe(chunk[i : i + 20]):
                tripped = True
                break
        self.assertTrue(tripped)

    def test_trace_fields(self) -> None:
        result = detect_output_degeneration(_LOG_DEGENERATE_TAIL)
        fields = result.trace_fields()
        self.assertTrue(fields["output_degeneration_unreliable"])
        self.assertIn("output_degeneration_components", fields)


if __name__ == "__main__":
    unittest.main()
