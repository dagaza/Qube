from __future__ import annotations

import unittest

from core.redacted_thinking_filter import RedactedThinkingStreamFilter


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


if __name__ == "__main__":
    unittest.main()
