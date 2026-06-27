"""Static built-in skill registry."""

from __future__ import annotations

from typing import Iterable

from core.skills.builtin.consumer_buying import CONSUMER_BUYING
from core.skills.builtin.debate_critical_thinking import DEBATE_CRITICAL_THINKING
from core.skills.builtin.decision_analysis import DECISION_ANALYSIS
from core.skills.builtin.interview_preparation import INTERVIEW_PREPARATION
from core.skills.builtin.learning_coach import LEARNING_COACH
from core.skills.builtin.meeting_processor import MEETING_PROCESSOR
from core.skills.builtin.memory_reflection import MEMORY_REFLECTION
from core.skills.builtin.optional import (
    CALENDAR_TASKS,
    CREATIVE_WRITING,
    DATA_INTERPRETATION,
)
from core.skills.builtin.problem_solving import PROBLEM_SOLVING
from core.skills.builtin.productivity_planning import PRODUCTIVITY_PLANNING
from core.skills.builtin.prompt_engineering import PROMPT_ENGINEERING
from core.skills.builtin.research_synthesis import RESEARCH_SYNTHESIS
from core.skills.builtin.scientific_research import SCIENTIFIC_RESEARCH
from core.skills.builtin.software_engineering import SOFTWARE_ENGINEERING
from core.skills.builtin.socratic_tutor import SOCRATIC_TUTOR
from core.skills.builtin.task_decomposition import TASK_DECOMPOSITION
from core.skills.builtin.writing_assistance import WRITING_ASSISTANCE
from core.skills.types import Skill

_BUILTIN_SKILLS: tuple[Skill, ...] = (
    TASK_DECOMPOSITION,
    PROBLEM_SOLVING,
    SOFTWARE_ENGINEERING,
    DECISION_ANALYSIS,
    PRODUCTIVITY_PLANNING,
    MEETING_PROCESSOR,
    PROMPT_ENGINEERING,
    RESEARCH_SYNTHESIS,
    SCIENTIFIC_RESEARCH,
    DEBATE_CRITICAL_THINKING,
    CONSUMER_BUYING,
    WRITING_ASSISTANCE,
    SOCRATIC_TUTOR,
    LEARNING_COACH,
    MEMORY_REFLECTION,
    INTERVIEW_PREPARATION,
    CALENDAR_TASKS,
    DATA_INTERPRETATION,
    CREATIVE_WRITING,
)

_REGISTRY: dict[str, Skill] = {s.id: s for s in _BUILTIN_SKILLS}


def iter_skills() -> Iterable[Skill]:
    return _REGISTRY.values()


def get_skill(skill_id: str) -> Skill | None:
    return _REGISTRY.get(skill_id)


def register_skill(skill: Skill, *, replace: bool = False) -> None:
    """Test-only registration hook."""
    if skill.id in _REGISTRY and not replace:
        raise ValueError(f"Skill already registered: {skill.id}")
    _REGISTRY[skill.id] = skill


def reset_registry_for_tests() -> None:
    global _REGISTRY
    _REGISTRY = {s.id: s for s in _BUILTIN_SKILLS}
