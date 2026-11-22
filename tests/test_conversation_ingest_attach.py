import unittest

from core.composer_attachments import ComposerAttachment
from core.composer_draft import ComposerDraft
from core.conversation_ingest_attach import replace_primary_routing_with_file


class TestConversationIngestAttach(unittest.TestCase):
    def test_replace_primary_routing_with_file_empty(self):
        draft = ComposerDraft()
        updated, added = replace_primary_routing_with_file(draft, "doc.pdf")
        self.assertTrue(added)
        self.assertEqual(len(updated.routing), 1)
        self.assertEqual(updated.routing[0].id, "doc.pdf")

    def test_replace_primary_routing_with_file_replaces_tool(self):
        draft = ComposerDraft(
            routing=[
                ComposerAttachment(kind="tool", id="library", label="Library"),
            ]
        )
        updated, added = replace_primary_routing_with_file(draft, "doc.pdf")
        self.assertTrue(added)
        self.assertEqual(len(updated.routing), 1)
        self.assertEqual(updated.routing[0].kind, "file")


if __name__ == "__main__":
    unittest.main()