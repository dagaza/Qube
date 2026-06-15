"""Tests for structured composer draft state."""

import unittest

from core.composer_attachments import ComposerAttachment
from core.composer_draft import (
    ROUTING_REJECT_ONE_SOURCE,
    ComposerDraft,
    add_routing_attachment,
    add_skill,
    composer_one_source_limit_request,
    draft_from_text,
    merge_drafts,
    remove_routing_at,
    serialize_draft,
    skill_chip_tooltip,
)
from core.composer_skills import ComposerSkillMention


class TestComposerDraft(unittest.TestCase):
    def test_serialize_orders_skills_before_routing_before_body(self) -> None:
        draft = ComposerDraft(
            body="Summarize please",
            routing=[
                ComposerAttachment(kind="tool", id="library", label="Library"),
            ],
            skills=[
                ComposerSkillMention(id="research_synthesis", label="Research Synthesis"),
            ],
        )
        raw = serialize_draft(draft)
        self.assertIn("@[skill:research_synthesis]", raw)
        self.assertIn("@[tool:library]", raw)
        self.assertIn("Summarize please", raw)
        self.assertLess(raw.index("@[skill:"), raw.index("@[tool:"))
        self.assertLess(raw.index("@[tool:"), raw.index("Summarize"))

    def test_draft_from_text_round_trip(self) -> None:
        text = (
            "@[skill:decision_analysis] @[file:notes.pdf] "
            "@[tool:library] What changed?"
        )
        draft = draft_from_text(text)
        self.assertEqual(draft.body, "What changed?")
        self.assertEqual(len(draft.skills), 1)
        self.assertEqual(draft.skills[0].id, "decision_analysis")
        self.assertEqual(len(draft.routing), 2)
        self.assertEqual(draft.routing[0].kind, "file")
        self.assertEqual(draft.routing[1].id, "library")

    def test_add_routing_allows_only_one_source(self) -> None:
        draft = ComposerDraft(
            routing=[
                ComposerAttachment(kind="tool", id="library", label="Library"),
            ]
        )
        memory = ComposerAttachment(kind="tool", id="memory", label="Memory")
        updated, added, reason = add_routing_attachment(draft, memory)
        self.assertFalse(added)
        self.assertEqual(reason, ROUTING_REJECT_ONE_SOURCE)
        self.assertEqual(len(updated.routing), 1)
        self.assertEqual(updated.routing[0].id, "library")

        same, added_again, reason_again = add_routing_attachment(updated, memory)
        self.assertFalse(added_again)
        self.assertEqual(reason_again, ROUTING_REJECT_ONE_SOURCE)
        self.assertEqual(len(same.routing), 1)

    def test_add_routing_dedupes_same_attachment(self) -> None:
        draft = ComposerDraft(
            routing=[
                ComposerAttachment(kind="tool", id="library", label="Library"),
            ]
        )
        same, added, reason = add_routing_attachment(draft, draft.routing[0])
        self.assertFalse(added)
        self.assertIsNone(reason)
        self.assertEqual(len(same.routing), 1)

    def test_add_routing_skips_internet_when_web_active(self) -> None:
        draft = ComposerDraft()
        internet = ComposerAttachment(kind="tool", id="internet", label="Internet")
        updated, added, reason = add_routing_attachment(
            draft,
            internet,
            skip_internet_when_web_active=True,
        )
        self.assertFalse(added)
        self.assertIsNone(reason)
        self.assertEqual(updated.routing, [])

    def test_add_skill_dedupes(self) -> None:
        draft = ComposerDraft()
        mention = ComposerSkillMention(id="problem_solving", label="Problem Solving")
        updated, added = add_skill(draft, mention)
        self.assertTrue(added)
        same, added_again = add_skill(updated, mention)
        self.assertFalse(added_again)
        self.assertEqual(len(same.skills), 1)

    def test_merge_drafts_preserves_existing_routing(self) -> None:
        base = ComposerDraft(
            body="old",
            routing=[
                ComposerAttachment(kind="file", id="a.pdf", label="a.pdf"),
            ],
            skills=[
                ComposerSkillMention(id="research_synthesis", label="Research Synthesis"),
            ],
        )
        lifted = draft_from_text("@[tool:memory] @[skill:decision_analysis] new body")
        merged, reject_reason = merge_drafts(base, lifted)
        self.assertEqual(reject_reason, ROUTING_REJECT_ONE_SOURCE)
        self.assertEqual(merged.body, "new body")
        self.assertEqual([a.id for a in merged.routing], ["a.pdf"])
        self.assertEqual(
            [s.id for s in merged.skills],
            ["research_synthesis", "decision_analysis"],
        )

    def test_merge_drafts_adds_first_lifted_routing_when_base_empty(self) -> None:
        base = ComposerDraft(body="keep")
        lifted = draft_from_text("@[tool:memory] @[tool:library] question")
        merged, reject_reason = merge_drafts(base, lifted)
        self.assertEqual(reject_reason, ROUTING_REJECT_ONE_SOURCE)
        self.assertEqual(merged.body, "question")
        self.assertEqual(len(merged.routing), 1)
        self.assertEqual(merged.routing[0].id, "memory")

    def test_composer_one_source_limit_request(self) -> None:
        req = composer_one_source_limit_request()
        self.assertTrue(req.show_countdown)
        self.assertEqual(req.auto_dismiss_ms, 5000)
        self.assertEqual(req.dedupe_key, "composer_one_source_limit")

    def test_skill_chip_tooltip_uses_registry_description(self) -> None:
        mention = ComposerSkillMention(id="decision_analysis", label="Decision Analysis")
        tip = skill_chip_tooltip(mention)
        self.assertIn("Decision Analysis", tip)
        self.assertNotIn("AttributeError", tip)

    def test_remove_routing_at(self) -> None:
        draft = ComposerDraft(
            routing=[
                ComposerAttachment(kind="file", id="a.pdf", label="a.pdf"),
            ]
        )
        updated = remove_routing_at(draft, 0)
        self.assertEqual(len(updated.routing), 0)


if __name__ == "__main__":
    unittest.main()
