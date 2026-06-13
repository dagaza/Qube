"""Decision analysis skill."""

from __future__ import annotations

from core.skills.base import BuiltinSkill


DECISION_ANALYSIS = BuiltinSkill(
    id="decision_analysis",
    name="Decision analysis",
    description="Decision matrices, risk analysis, and reversible vs irreversible choices.",
    version="1.0.0",
    priority=82,
    max_prompt_chars=420,
    activation_triggers=(
        "should i",
        "which option",
        "pros and cons",
        "pros & cons",
        "decide between",
        "decision matrix",
        "opportunity cost",
        "reversible",
        "irreversible",
        "weigh the options",
        "help me choose",
    ),
    prompt_fragment=(
        "Frame the decision: options → criteria → weighted scores or pros/cons table → "
        "risks and opportunity costs → note reversible vs irreversible consequences → "
        "clear recommendation with confidence level. Do not invent facts about options."
    ),
)
