"""Tests for composer @-mention attachment tokens."""

import unittest

from core.composer_attachments import (
    ComposerAttachment,
    composer_tool_tooltip,
    composer_tools_for_palette,
    format_token,
    parse_attachments,
    resolve_attachment_routing,
    strip_tokens_for_display,
    validate_file_token,
)


class TestComposerAttachments(unittest.TestCase):
    def test_format_and_parse_file(self):
        att = ComposerAttachment(kind="file", id="Omega.pdf", label="Omega.pdf")
        token = format_token(att)
        self.assertEqual(token, "@[file:Omega.pdf]")
        clean, attachments = parse_attachments(f"Summarize {token} please")
        self.assertEqual(clean, "Summarize please")
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].kind, "file")
        self.assertEqual(attachments[0].id, "Omega.pdf")

    def test_parse_conversation_and_tool(self):
        text = "@[chat:abc-123] @[tool:internet] hello"
        clean, attachments = parse_attachments(text)
        self.assertEqual(clean, "hello")
        self.assertEqual(len(attachments), 2)
        self.assertEqual(attachments[0].kind, "conversation")
        self.assertEqual(attachments[0].id, "abc-123")
        self.assertEqual(attachments[1].kind, "tool")
        self.assertEqual(attachments[1].id, "internet")

    def test_format_conversation_token_shows_title(self):
        att = ComposerAttachment(
            kind="conversation",
            id="068021a6-4660-40ff-bf13-1949116eca11",
            label="Is Brown Rice Good for Cooking",
        )
        token = format_token(att)
        self.assertIn("Is Brown Rice Good for Cooking", token)
        self.assertIn("068021a6-4660-40ff-bf13-1949116eca11", token)
        clean, attachments = parse_attachments(f"{token} summarize")
        self.assertEqual(attachments[0].id, "068021a6-4660-40ff-bf13-1949116eca11")
        self.assertEqual(attachments[0].label, "Is Brown Rice Good for Cooking")

    def test_strip_tokens(self):
        raw = "@[file:a.pdf] What is X?"
        self.assertEqual(strip_tokens_for_display(raw), "What is X?")

    def test_strip_skill_tokens(self):
        raw = "@[skill:decision_analysis] What is X?"
        self.assertEqual(strip_tokens_for_display(raw), "What is X?")

    def test_validate_file_token_rejects_bracket(self):
        self.assertFalse(validate_file_token("bad].pdf"))
        self.assertTrue(validate_file_token("good.pdf"))

    def test_resolve_file_routing(self):
        att = ComposerAttachment(kind="file", id="doc.txt", label="doc.txt")
        patch = resolve_attachment_routing([att])
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch["route"], "rag")
        self.assertEqual(patch["source_filter"], "doc.txt")
        self.assertIn("composer_attachments", patch)

    def test_resolve_conversation_routing(self):
        att = ComposerAttachment(kind="conversation", id="sess-1", label="Title")
        patch = resolve_attachment_routing([att])
        assert patch is not None
        self.assertEqual(patch["route"], "none")
        self.assertEqual(patch["referenced_session_id"], "sess-1")

    def test_resolve_tool_library(self):
        att = ComposerAttachment(kind="tool", id="library", label="Library")
        patch = resolve_attachment_routing([att])
        assert patch is not None
        self.assertEqual(patch["route"], "web")
        self.assertEqual(patch["strategy"], "attachment_tool_library")

    def test_resolve_tool_research(self):
        att = ComposerAttachment(kind="tool", id="research", label="Deep research")
        patch = resolve_attachment_routing([att])
        assert patch is not None
        self.assertEqual(patch["route"], "deep_research")
        self.assertEqual(patch["strategy"], "attachment_tool_research")

    def test_resolve_tool_fetch_routes_web(self):
        att = ComposerAttachment(kind="tool", id="fetch", label="Fetch")
        patch = resolve_attachment_routing([att])
        assert patch is not None
        self.assertEqual(patch["route"], "web")
        self.assertEqual(patch["attachment_tool"], "fetch")

    def test_palette_includes_fetch_and_hides_recipe_by_default(self):
        ids = [str(t["id"]) for t in composer_tools_for_palette("")]
        self.assertIn("fetch", ids)
        self.assertNotIn("recipe", ids)

    def test_palette_shows_recipe_when_id_matches(self):
        ids = [str(t["id"]) for t in composer_tools_for_palette("recipe")]
        self.assertEqual(ids, ["recipe"])

    def test_palette_hides_science_alias_by_default(self):
        ids = [str(t["id"]) for t in composer_tools_for_palette("")]
        self.assertIn("evidence", ids)
        self.assertNotIn("science", ids)

    def test_palette_shows_science_when_id_matches(self):
        ids = [str(t["id"]) for t in composer_tools_for_palette("science")]
        self.assertEqual(ids, ["science"])

    def test_palette_scientific_shows_evidence_not_science(self):
        ids = [str(t["id"]) for t in composer_tools_for_palette("scientific")]
        self.assertEqual(ids, ["evidence"])

    def test_composer_tool_tooltip_includes_label_and_token(self):
        tool = next(t for t in composer_tools_for_palette("") if t["id"] == "evidence")
        tip = composer_tool_tooltip(tool)
        self.assertIn("Scientific literature", tip)
        self.assertIn("@[tool:evidence]", tip)


if __name__ == "__main__":
    unittest.main()
