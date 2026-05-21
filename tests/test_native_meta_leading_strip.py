from __future__ import annotations

import unittest

from core.native_meta_leading_strip import LeadingMetaInstructionStripper


class TestLeadingMetaInstructionStripper(unittest.TestCase):
    def test_strips_provide_final_answer_prefix_and_keeps_answer(self) -> None:
        f = LeadingMetaInstructionStripper()
        out = f.feed("Provide final answer ")
        out += f.feed("The sky is blue because air molecules scatter blue light.")
        out += f.flush()
        self.assertEqual(
            out,
            "The sky is blue because air molecules scatter blue light.",
        )

    def test_holds_exact_provide_final_answer_until_answer_arrives(self) -> None:
        f = LeadingMetaInstructionStripper()
        self.assertEqual(f.feed("Provide final"), "")
        self.assertEqual(f.feed(" answer"), "")
        out = f.feed("\nRayleigh scattering makes the sky look blue.")
        out += f.flush()
        self.assertEqual(out, "Rayleigh scattering makes the sky look blue.")

    def test_strips_completed_provide_final_answer_sentence(self) -> None:
        f = LeadingMetaInstructionStripper()
        self.assertEqual(f.feed("Provide final answer."), "")
        out = f.feed(" The sky is blue because molecules scatter blue light.")
        out += f.flush()
        self.assertEqual(out, "The sky is blue because molecules scatter blue light.")

    def test_lone_provide_final_answer_flushes_to_empty(self) -> None:
        f = LeadingMetaInstructionStripper()
        self.assertEqual(f.feed("Provide final answer."), "")
        self.assertEqual(f.flush(), "")

    def test_strips_quoted_provide_final_answer_prefix(self) -> None:
        f = LeadingMetaInstructionStripper()
        self.assertEqual(f.feed('"Provide final'), "")
        self.assertEqual(f.feed(' answer."'), "")
        out = f.feed("\nThe sky is blue because air scatters short wavelengths.")
        out += f.flush()
        self.assertEqual(out, "The sky is blue because air scatters short wavelengths.")

    def test_holds_newline_split_provide_final_answer(self) -> None:
        f = LeadingMetaInstructionStripper()
        self.assertEqual(f.feed("Provide final\n"), "")
        self.assertEqual(f.feed("answer."), "")
        out = f.feed("\nRayleigh scattering makes the sky look blue.")
        out += f.flush()
        self.assertEqual(out, "Rayleigh scattering makes the sky look blue.")


if __name__ == "__main__":
    unittest.main()
