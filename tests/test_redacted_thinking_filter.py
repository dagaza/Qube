from __future__ import annotations

import unittest

from core.redacted_thinking_filter import (
    RedactedThinkingStreamFilter,
    strip_reasoning_blocks_from_text,
)


class TestRedactedThinkingStreamFilter(unittest.TestCase):
    def test_strips_redacted_thinking_block(self) -> None:
        f = RedactedThinkingStreamFilter()
        out = f.feed("final before <redacted_thinking>hidden</redacted_thinking> final after")
        out += f.flush()
        self.assertEqual(out, "final before  final after")

    def test_strips_think_block_split_across_chunks(self) -> None:
        f = RedactedThinkingStreamFilter()
        parts = [
            "The answer is ",
            "<thi",
            "nk>internal plan",
            "</th",
            "ink>Rayleigh scattering.",
        ]
        out = "".join(f.feed(p) for p in parts) + f.flush()
        self.assertEqual(out, "The answer is Rayleigh scattering.")

    def test_strips_thinking_block(self) -> None:
        f = RedactedThinkingStreamFilter()
        out = f.feed("<thinking>Provide brief explanation.</thinking>Blue light scatters.")
        out += f.flush()
        self.assertEqual(out, "Blue light scatters.")


class TestStripReasoningBlocksFromText(unittest.TestCase):
    def test_strips_qwen3_think_block_multiline(self) -> None:
        raw = (
            "<Think>\nUser asked about weather.\nPick a short label.\n</Think>\n"
            "Copenhagen Weather"
        )
        self.assertEqual(
            strip_reasoning_blocks_from_text(raw).strip(),
            "Copenhagen Weather",
        )

    def test_strips_lowercase_thinking_block(self) -> None:
        raw = "<thinking>plan</thinking>\nSky Color"
        self.assertEqual(strip_reasoning_blocks_from_text(raw).strip(), "Sky Color")

    def test_strips_unclosed_think_tail(self) -> None:
        raw = "<Think>\npartial reasoning\n"
        self.assertEqual(strip_reasoning_blocks_from_text(raw).strip(), "")


if __name__ == "__main__":
    unittest.main()
