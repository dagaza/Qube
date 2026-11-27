"""Plan S41 (V67, D46): the memory proper-noun gate takes names, not sentence starters.

When a query holds a capitalised token that is not a stopword, memory search
keeps only memories containing it. Sentence starters with an apostrophe or
outside the stopword list ("What's", "Am", "Write", "Help", "Recommend",
"Explain") and the title "Dr" counted as names, so "What's my cat called?"
dropped every memory. Now the first word of each sentence and titles are
skipped, and possessives and contractions are stripped.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.memory_tool import _extract_query_proper_nouns as nouns  # noqa: E402

# Fail on the old extraction.
SENTENCE_STARTERS_AND_TITLES = {
    "What's my cat called?": [],
    "Am I allergic to anything?": [],
    "What's my car's registration number?": [],
    "Write a short note to my landlord asking if I can repaint the bedroom.": [],
    "Help me plan a calmer evening routine.": [],
    "Recommend something to listen to on my commute.": [],
    "Explain how a hash map works.": [],
    "Tell me about Dr. Evelyn.": ["Evelyn"],
    "What's the capital of Australia?": ["Australia"],
    "Did I tell you about Kofi's party?": ["Kofi"],
}

# Unchanged.
NAMES_KEPT = {
    "Remind me who my master's supervisor was.": [],
    "Tell me about Dr.Evelyn": ["Evelyn"],
    "Is Tomasz coming to the Lisbon trip?": ["Tomasz", "Lisbon"],
    "I met Kofi. Kofi said hi.": ["Kofi"],
    "Can you remind me what Qube is?": [],
    "": [],
}


class TestProperNounExtraction(unittest.TestCase):
    def test_sentence_starters_titles_and_possessives(self):
        for query, expected in SENTENCE_STARTERS_AND_TITLES.items():
            with self.subTest(query=query):
                self.assertEqual(nouns(query), expected)

    def test_names_inside_a_sentence_are_kept(self):
        for query, expected in NAMES_KEPT.items():
            with self.subTest(query=query):
                self.assertEqual(nouns(query), expected)

    def test_a_name_opening_the_question_is_not_required(self):
        # The known cost: the gate cannot tell "Kofi's birthday?" from "What's ...?".
        self.assertEqual(nouns("Kofi's birthday?"), [])


if __name__ == "__main__":
    unittest.main()
