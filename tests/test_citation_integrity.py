"""Unit tests for citation integrity detection and repair."""
from __future__ import annotations

import unittest

from core.citation_integrity import (
    analyze_citations,
    extract_citation_tokens,
    find_orphan_citations,
    is_missing_citation_exempt,
    missing_web_citation,
    repair_orphan_citations,
    valid_source_ids,
)
from core.citation_normalize import normalize_combined_numeric_citations


class CitationIntegrityTests(unittest.TestCase):
    def test_single_web_valid_w(self) -> None:
        sources = [{"id": "W", "type": "web", "filename": "hit"}]
        report = analyze_citations("It is sunny [W].", sources)
        self.assertFalse(report.has_violation)
        self.assertEqual(report.orphan_ids, ())

    def test_multi_web_valid_numeric(self) -> None:
        sources = [
            {"id": 1, "type": "web"},
            {"id": 2, "type": "web"},
        ]
        report = analyze_citations("First point [1]. Second [2].", sources)
        self.assertFalse(report.has_violation)

    def test_multi_web_invalid_w(self) -> None:
        sources = [{"id": 1, "type": "web"}, {"id": 2, "type": "web"}]
        report = analyze_citations("1. [W] A summary statement.", sources)
        self.assertTrue(report.has_violation)
        self.assertIn("W", report.orphan_ids)

    def test_multi_web_invalid_high_numeric(self) -> None:
        sources = [{"id": 1, "type": "web"}, {"id": 2, "type": "web"}]
        report = analyze_citations("See [3] for detail.", sources)
        self.assertEqual(report.orphan_ids, ("3",))

    def test_no_sources_orphan_numeric(self) -> None:
        report = analyze_citations("Answer [1].", [])
        self.assertTrue(report.has_violation)
        self.assertEqual(report.orphan_ids, ("1",))

    def test_combined_numeric_citations_detected(self) -> None:
        text = normalize_combined_numeric_citations("Score [1, 2].")
        tokens = extract_citation_tokens(text)
        self.assertEqual(tokens, {"1", "2"})

    def test_find_orphan_citations(self) -> None:
        orphans = find_orphan_citations(
            "Bad [W] cite",
            [{"id": 1}, {"id": 2}],
        )
        self.assertEqual(orphans, ["W"])

    def test_valid_source_ids_normalizes_int(self) -> None:
        ids = valid_source_ids([{"id": 1}, {"id": "2"}])
        self.assertEqual(ids, {"1", "2"})

    def test_repair_strip_removes_orphans(self) -> None:
        sources = [{"id": 1}, {"id": 2}]
        repaired, post = repair_orphan_citations(
            "1. [W] Summary here.",
            sources,
            mode="strip",
        )
        self.assertFalse(post.has_violation)
        self.assertNotIn("[W]", repaired)

    def test_repair_plain_unchanged(self) -> None:
        text = "1. [W] Summary."
        sources = [{"id": 1}, {"id": 2}]
        out, report = repair_orphan_citations(text, sources, mode="plain")
        self.assertEqual(out, text)
        self.assertTrue(report.has_violation)

    def test_missing_web_citation_when_sources_present(self) -> None:
        sources = [
            {"id": 1, "type": "web"},
            {"id": 2, "type": "web"},
            {"id": 3, "type": "web"},
        ]
        answer = (
            "South Korea defeated Czechia, and Mexico beat South Africa "
            "in the first day of games for the World Cup 2026."
        )
        report = analyze_citations(answer, sources)
        self.assertFalse(report.has_violation)
        self.assertTrue(report.missing_citation)
        self.assertTrue(report.has_citation_issue)
        self.assertTrue(missing_web_citation(answer, cited_ids=set(), web_hit_count=3))

    def test_missing_citation_exempt_disclaimer(self) -> None:
        sources = [{"id": 1, "type": "web"}]
        answer = "None of the sources are relevant to your question."
        report = analyze_citations(answer, sources)
        self.assertFalse(report.missing_citation)
        self.assertTrue(is_missing_citation_exempt(answer))

    def test_missing_citation_not_flagged_without_web_hits(self) -> None:
        sources = [{"id": 1, "type": "rag"}]
        report = analyze_citations("Answer without brackets.", sources)
        self.assertFalse(report.missing_citation)


if __name__ == "__main__":
    unittest.main()
