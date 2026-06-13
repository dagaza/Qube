"""Orthogonal reasoning-skills layer (prompt injection only; never routes)."""

from core.skills.activation import activate_skills
from core.skills.context import build_skill_context
from core.skills.registry import get_skill, iter_skills, register_skill
from core.skills.types import (
    SkillActivation,
    SkillActivationResult,
    SkillContext,
    SkillSettings,
)

__all__ = [
    "SkillActivation",
    "SkillActivationResult",
    "SkillContext",
    "SkillSettings",
    "activate_skills",
    "build_skill_context",
    "get_skill",
    "iter_skills",
    "register_skill",
]
