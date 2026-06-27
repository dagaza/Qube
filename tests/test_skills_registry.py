"""Registry and context builder tests."""

from __future__ import annotations

import unittest

from core.skills.context import build_skill_context
from core.skills.registry import get_skill, iter_skills, reset_registry_for_tests
from core.skills.types import SkillSettings
from core.skills.activation import activate_skills


class SkillRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_registry_for_tests()

    def test_builtin_skills_registered(self) -> None:
        ids = {s.id for s in iter_skills()}
        self.assertIn("task_decomposition", ids)
        self.assertIn("software_engineering", ids)
        self.assertIn("problem_solving", ids)
        self.assertIn("prompt_engineering", ids)
        self.assertIn("scientific_research", ids)
        self.assertEqual(len(ids), 19)

    def test_get_skill(self) -> None:
        skill = get_skill("task_decomposition")
        self.assertIsNotNone(skill)
        assert skill is not None
        self.assertEqual(skill.name, "Task decomposition")


class SkillContextTests(unittest.TestCase):
    def test_build_skill_context_read_only(self) -> None:
        ctx = build_skill_context(
            user_query="Hello",
            clean_query="hello",
            execution_route="NONE",
            all_ui_sources=[],
            follow_up_active=False,
            explicit_remember_active=False,
            file_search_active=False,
            narrative_active=False,
            decision={"top_intent": "chat", "trace": {"winning_reason": "none"}},
        )
        self.assertEqual(ctx.execution_route, "NONE")
        self.assertEqual(ctx.router_top_intent, "chat")
        self.assertEqual(ctx.router_trace_summary, "none")


class SkillActivationDisabledTests(unittest.TestCase):
    def test_disabled_returns_empty(self) -> None:
        ctx = build_skill_context(
            user_query="break down my project plan step by step",
            clean_query="break down my project plan step by step",
            execution_route="NONE",
            all_ui_sources=[],
            follow_up_active=False,
            explicit_remember_active=False,
            file_search_active=False,
            narrative_active=False,
        )
        result = activate_skills(ctx, settings=SkillSettings(enabled=False))
        self.assertEqual(result.activations, ())
        self.assertEqual(result.skipped_reason, "disabled")


if __name__ == "__main__":
    unittest.main()
