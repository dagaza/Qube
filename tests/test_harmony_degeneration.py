"""Regression tests for gpt-oss degeneration truncation (log-derived)."""
from __future__ import annotations

import unittest

from core.harmony_degeneration import (
    find_degeneration_start,
    polish_harmony_visible_text,
    truncate_at_degeneration,
)
from core.harmony_stream_parser import HarmonyStreamParser
from core.output_artifact_strip import strip_harmony_oss_artifacts


_LOG_DEGENERATE_TAIL = (
    "Birds take to‑bath in a few different ways, and the reasons are pretty practical.  \n"
    "1. **Cleaning** – The most obvious reason is to keep their feathers clean and free "
    "from dirt, parasites, and oil‑rich prey. A wet wash or a dust‑bathing shower can strip "
    "away excess oil, “muss”‑like‑mucus, and the buildup of bacteria or fungi that often, "
    "especially after a long flight or one‑to‑one.  \n\n"
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


class TestHarmonyDegeneration(unittest.TestCase):
    def test_valid_second_list_item_not_truncated(self) -> None:
        self.assertIsNone(find_degeneration_start(_VALID_TWO_POINT))

    def test_orphan_list_fragment(self) -> None:
        from core.harmony_degeneration import is_harmony_orphan_stream_fragment

        self.assertTrue(is_harmony_orphan_stream_fragment("2"))
        self.assertTrue(is_harmony_orphan_stream_fragment("\n\n2."))
        self.assertFalse(is_harmony_orphan_stream_fragment("2. **Heat** –"))

    def test_finds_broken_second_list_item(self) -> None:
        cut = find_degeneration_start(_LOG_DEGENERATE_TAIL)
        self.assertIsNotNone(cut)
        assert cut is not None
        self.assertIn("2. **Pre", _LOG_DEGENERATE_TAIL[cut : cut + 20])

    def test_truncate_keeps_first_section_only(self) -> None:
        out = truncate_at_degeneration(_LOG_DEGENERATE_TAIL)
        self.assertIn("1. **Cleaning**", out)
        self.assertNotIn("2. **Pre", out)
        self.assertNotIn("We……", out)

    def test_trim_abrupt_generation_end_drops_partial_tail(self) -> None:
        raw = (
            "Kathmandu is home to religious minorities, including Muslims and Christians, "
            "and followers of indigenous traditions such as the Newar people’s worship of local deities. "
            "In addition, small pockets of Kirghiz add diversity, giving a vibrant, inclusive‑tied…"
        )
        out = polish_harmony_visible_text(raw)
        self.assertIn("local deities.", out)
        self.assertNotIn("Kirghiz", out)
        self.assertNotIn("…", out)
        p = HarmonyStreamParser()
        emitted = ""
        for i in range(0, len(_LOG_DEGENERATE_TAIL), 40):
            emitted += p.feed(_LOG_DEGENERATE_TAIL[i : i + 40])
        self.assertIn("1. **Cleaning**", emitted)
        self.assertNotIn("2. **Pre", emitted)
        self.assertNotIn("We……", emitted)
        self.assertTrue(p.degeneration_detected)

    def test_strip_uses_truncate(self) -> None:
        out = strip_harmony_oss_artifacts(_LOG_DEGENERATE_TAIL)
        self.assertNotIn("We……", out)

    def test_polish_drops_dangling_clause_tail(self) -> None:
        raw = (
            "Birds bathe to stay clean. Cleaning – Removes dirt from feathers, "
            "that often, especially after a long flight or one-to-one."
        )
        out = polish_harmony_visible_text(raw)
        self.assertIn("Removes dirt", out)
        self.assertNotIn("especially after", out)

    def test_pre_double_hyphen_still_detected(self) -> None:
        self.assertIsNotNone(
            find_degeneration_start("Intro.\n\n2. **Pre‑‑treatment –** ??")
        )
