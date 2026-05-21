"""Tests for LLMWorker composer attachment routing overrides."""

import unittest

from core.composer_attachments import (
    ComposerAttachment,
    parse_attachments,
    resolve_attachment_routing,
)


class TestLLMWorkerComposerRouting(unittest.TestCase):
    def test_file_attachment_forces_rag_with_source_filter(self):
        att = ComposerAttachment(kind="file", id="Notes.md", label="Notes.md")
        patch = resolve_attachment_routing([att])
        assert patch is not None
        self.assertEqual(patch["route"], "rag")
        self.assertTrue(patch.get("attachment_file"))
        self.assertEqual(patch["source_filter"], "Notes.md")
        self.assertEqual(patch["strategy"], "attachment_file")

    def test_memory_tool_forces_memory_route(self):
        att = ComposerAttachment(kind="tool", id="memory", label="Memory")
        patch = resolve_attachment_routing([att])
        assert patch is not None
        self.assertEqual(patch["route"], "memory")
        self.assertEqual(patch["attachment_tool"], "memory")

    def test_internet_tool_forces_web_route(self):
        att = ComposerAttachment(kind="tool", id="internet", label="Internet")
        patch = resolve_attachment_routing([att])
        assert patch is not None
        self.assertEqual(patch["route"], "web")
        self.assertEqual(patch["attachment_tool"], "internet")
        self.assertEqual(patch["strategy"], "attachment_tool_internet")

    def test_parse_tool_internet_token(self):
        clean, attachments = parse_attachments(
            "@[tool:internet] What is the cooking time for rice?"
        )
        self.assertEqual(clean, "What is the cooking time for rice?")
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].id, "internet")


if __name__ == "__main__":
    unittest.main()
