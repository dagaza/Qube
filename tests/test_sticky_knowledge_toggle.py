"""Sticky Web / Trusted / Evidence knowledge route toggles."""

import unittest

from core.composer_attachments import ComposerAttachment
from core.composer_draft import ComposerDraft, add_routing_attachment
from core.composer_sticky_tools import (
    composer_sticky_knowledge_switched_request,
    sticky_knowledge_worker_flags,
)


class StickyKnowledgeWorkerFlagTests(unittest.TestCase):
    def test_internet_sticky_enables_force_web_only(self) -> None:
        flags = sticky_knowledge_worker_flags("internet")
        self.assertEqual(flags.sticky_knowledge_tool, "internet")
        self.assertTrue(flags.force_web_enabled)
        self.assertIsNone(flags.force_composer_tool)

    def test_evidence_sticky_sets_composer_tool(self) -> None:
        flags = sticky_knowledge_worker_flags("evidence")
        self.assertEqual(flags.force_composer_tool, "evidence")
        self.assertTrue(flags.force_web_enabled)

    def test_clear_sticky_clears_flags(self) -> None:
        flags = sticky_knowledge_worker_flags(None)
        self.assertIsNone(flags.sticky_knowledge_tool)
        self.assertFalse(flags.force_web_enabled)


class StickyKnowledgeNotificationTests(unittest.TestCase):
    def test_switch_toast_mentions_modes(self) -> None:
        req = composer_sticky_knowledge_switched_request(
            previous_id="internet",
            new_id="evidence",
        )
        self.assertEqual(req.title, "One search mode per chat")
        self.assertIn("Web", req.body)
        self.assertIn("Evidence", req.body)
        self.assertEqual(req.dedupe_key, "composer_sticky_knowledge_exclusive")
        self.assertTrue(req.show_countdown)
        self.assertGreater(req.auto_dismiss_ms, 0)


class StickyKnowledgeComposerDraftTests(unittest.TestCase):
    def test_skips_evidence_chip_when_sticky_active(self) -> None:
        draft = ComposerDraft()
        evidence = ComposerAttachment(kind="tool", id="evidence", label="Evidence")
        updated, added, reason = add_routing_attachment(
            draft,
            evidence,
            skip_redundant_tools_when_sticky_knowledge=True,
        )
        self.assertFalse(added)
        self.assertIsNone(reason)
        self.assertEqual(updated.routing, [])


if __name__ == "__main__":
    unittest.main()