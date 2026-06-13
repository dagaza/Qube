"""Golden activation tests for built-in skills."""

from __future__ import annotations

import unittest

from core.skills.activation import activate_skills
from core.skills.context import build_skill_context
from core.skills.registry import reset_registry_for_tests
from core.skills.types import SkillSettings

_ENABLED = SkillSettings(enabled=True, min_activation_score=0.55, max_active_skills=3)


def _ctx(
    query: str,
    *,
    route: str = "NONE",
    sources: bool = False,
    follow_up: bool = False,
    remember: bool = False,
) -> object:
    return build_skill_context(
        user_query=query,
        clean_query=query,
        execution_route=route,
        all_ui_sources=[{"id": 1}] if sources else [],
        follow_up_active=follow_up,
        explicit_remember_active=remember,
        file_search_active=False,
        narrative_active=False,
    )


class SkillActivationGoldenTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_registry_for_tests()

    def test_task_decomposition_activates(self) -> None:
        result = activate_skills(
            _ctx("Can you break down this project step by step?"),
            settings=_ENABLED,
        )
        ids = [a.skill_id for a in result.activations]
        self.assertIn("task_decomposition", ids)
        self.assertIn("REASONING GUIDANCE", result.prompt_block)
        self.assertIn("[Task decomposition]", result.prompt_block)

    def test_software_engineering_activates_on_bug(self) -> None:
        result = activate_skills(
            _ctx("Help me debug this API error in my Python function"),
            settings=_ENABLED,
        )
        ids = [a.skill_id for a in result.activations]
        self.assertIn("software_engineering", ids)

    def test_writing_assistance_activates(self) -> None:
        result = activate_skills(
            _ctx("Please rewrite this email with a friendlier tone"),
            settings=_ENABLED,
        )
        ids = [a.skill_id for a in result.activations]
        self.assertIn("writing_assistance", ids)

    def test_compositional_multiple_skills(self) -> None:
        result = activate_skills(
            _ctx(
                "Break down how to debug and rewrite this draft email step by step"
            ),
            settings=_ENABLED,
        )
        self.assertGreaterEqual(len(result.activations), 2)

    def test_explicit_remember_skips(self) -> None:
        result = activate_skills(
            _ctx("remember that I like tea", remember=True),
            settings=_ENABLED,
        )
        self.assertEqual(result.skipped_reason, "explicit_remember")

    def test_schedule_today_does_not_force_web_skill(self) -> None:
        """Skills must not duplicate web routing heuristics."""
        result = activate_skills(
            _ctx("schedule my tasks for today"),
            settings=_ENABLED,
        )
        ids = [a.skill_id for a in result.activations]
        self.assertNotIn("research_synthesis", ids)

    def test_research_synthesis_boost_with_sources(self) -> None:
        result = activate_skills(
            _ctx("summarize the key findings from these notes", sources=True),
            settings=_ENABLED,
        )
        ids = [a.skill_id for a in result.activations]
        self.assertIn("research_synthesis", ids)
        research = next(a for a in result.activations if a.skill_id == "research_synthesis")
        self.assertTrue(any(s.startswith("boost:has_sources") for s in research.signals))

    def test_char_budget_enforced(self) -> None:
        tight = SkillSettings(
            enabled=True,
            min_activation_score=0.4,
            max_active_skills=5,
            total_prompt_char_budget=200,
        )
        result = activate_skills(
            _ctx(
                "Break down debug rewrite summarize compare step by step "
                "for my API bug draft email"
            ),
            settings=tight,
        )
        self.assertLessEqual(result.token_budget_applied, 200)

    def test_mutual_exclusion_technical_creative(self) -> None:
        result = activate_skills(
            _ctx("Write a story about debugging a Python API bug in my codebase"),
            settings=SkillSettings(
                enabled=True,
                min_activation_score=0.4,
                max_active_skills=5,
            ),
        )
        ids = [a.skill_id for a in result.activations]
        if "software_engineering" in ids and "creative_writing" in ids:
            self.fail("mutual exclusion group technical_creative violated")


class ExtendedSkillActivationTests(unittest.TestCase):
    """Golden tests for tier-1 and tier-2 extended skills."""

    def setUp(self) -> None:
        reset_registry_for_tests()

    def test_problem_solving_activates(self) -> None:
        result = activate_skills(
            _ctx(
                "What is the root cause and underlying issue? "
                "List tradeoffs before recommending a fix."
            ),
            settings=_ENABLED,
        )
        self.assertIn("problem_solving", [a.skill_id for a in result.activations])

    def test_decision_analysis_activates(self) -> None:
        result = activate_skills(
            _ctx("Should I take job A or job B? Help me weigh the options."),
            settings=_ENABLED,
        )
        self.assertIn("decision_analysis", [a.skill_id for a in result.activations])

    def test_meeting_processor_activates(self) -> None:
        result = activate_skills(
            _ctx("Here are my meeting notes — extract action items and who owns each."),
            settings=_ENABLED,
        )
        self.assertIn("meeting_processor", [a.skill_id for a in result.activations])

    def test_socratic_tutor_activates(self) -> None:
        result = activate_skills(
            _ctx("Help me understand recursion — don't give me the answer yet."),
            settings=_ENABLED,
        )
        self.assertIn("socratic_tutor", [a.skill_id for a in result.activations])

    def test_prompt_engineering_activates(self) -> None:
        result = activate_skills(
            _ctx("How should I ask the LLM to get better answers for code review?"),
            settings=_ENABLED,
        )
        self.assertIn("prompt_engineering", [a.skill_id for a in result.activations])

    def test_debate_critical_thinking_activates(self) -> None:
        result = activate_skills(
            _ctx("Play devil's advocate and steelman both sides of this claim."),
            settings=_ENABLED,
        )
        self.assertIn(
            "debate_critical_thinking",
            [a.skill_id for a in result.activations],
        )

    def test_learning_coach_activates(self) -> None:
        result = activate_skills(
            _ctx("Build me a study plan with practice problems for linear algebra."),
            settings=_ENABLED,
        )
        self.assertIn("learning_coach", [a.skill_id for a in result.activations])

    def test_consumer_buying_activates(self) -> None:
        result = activate_skills(
            _ctx("Which laptop should I buy? Compare models for long-term cost."),
            settings=_ENABLED,
        )
        self.assertIn("consumer_buying", [a.skill_id for a in result.activations])

    def test_interview_preparation_activates(self) -> None:
        result = activate_skills(
            _ctx("Mock interview: give me a behavioral question using the STAR method."),
            settings=_ENABLED,
        )
        self.assertIn(
            "interview_preparation",
            [a.skill_id for a in result.activations],
        )

    def test_decision_and_consumer_compose(self) -> None:
        result = activate_skills(
            _ctx(
                "Should I buy laptop X or Y? Product comparison with pros and cons "
                "and long-term cost."
            ),
            settings=SkillSettings(
                enabled=True,
                min_activation_score=0.55,
                max_active_skills=5,
            ),
        )
        ids = {a.skill_id for a in result.activations}
        self.assertIn("decision_analysis", ids)
        self.assertIn("consumer_buying", ids)

    def test_registry_has_eighteen_skills(self) -> None:
        from core.skills.registry import iter_skills

        self.assertEqual(len(list(iter_skills())), 18)

    def test_forced_skill_bypasses_disabled_setting(self) -> None:
        result = activate_skills(
            _ctx("hello"),
            settings=SkillSettings(enabled=False),
            forced_skill_ids=("decision_analysis",),
        )
        ids = [a.skill_id for a in result.activations]
        self.assertIn("decision_analysis", ids)
        self.assertEqual(result.forced_skill_ids, ("decision_analysis",))
        self.assertIn("forced:composer", result.activations[0].signals)

    def test_forced_skill_in_prompt_block(self) -> None:
        result = activate_skills(
            _ctx("hello"),
            settings=_ENABLED,
            forced_skill_ids=("socratic_tutor",),
        )
        self.assertIn("[Socratic tutor]", result.prompt_block)
        self.assertIn("skills_forced", result.telemetry_dict())

    def test_forced_plus_auto_respects_max_active(self) -> None:
        result = activate_skills(
            _ctx(
                "Break down debug rewrite summarize compare step by step "
                "for my API bug draft email"
            ),
            settings=SkillSettings(
                enabled=True,
                min_activation_score=0.4,
                max_active_skills=3,
            ),
            forced_skill_ids=("prompt_engineering",),
        )
        self.assertLessEqual(len(result.activations), 3)
        self.assertEqual(result.activations[0].skill_id, "prompt_engineering")


class SkillRoutingOrthogonalityTests(unittest.TestCase):
    """Skills must not change execution_route (read-only context)."""

    def test_route_unchanged_in_context(self) -> None:
        for route in ("NONE", "MEMORY", "RAG", "HYBRID", "WEB"):
            ctx = build_skill_context(
                user_query="break down this bug step by step",
                clean_query="break down this bug step by step",
                execution_route=route,
                all_ui_sources=[],
                follow_up_active=False,
                explicit_remember_active=False,
                file_search_active=False,
                narrative_active=False,
            )
            self.assertEqual(ctx.execution_route, route)
            result = activate_skills(ctx, settings=_ENABLED)
            self.assertEqual(ctx.execution_route, route)
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
