"""Optional Phase-4 skills: calendar, data interpretation, creative writing."""

from __future__ import annotations

import re

from core.skills.base import BuiltinSkill

CALENDAR_TASKS = BuiltinSkill(
    id="calendar_tasks",
    name="Calendar tasks",
    description="Action items with dates and reminders framing.",
    version="1.0.0",
    priority=60,
    max_prompt_chars=320,
    activation_triggers=(
        "meeting",
        "remind",
        "calendar",
        "by friday",
        "by monday",
        "appointment",
        "event",
    ),
    prompt_fragment=(
        "When scheduling: confirm date/time assumptions, list concrete action items "
        "with deadlines, and flag conflicts or missing details."
    ),
)

DATA_INTERPRETATION = BuiltinSkill(
    id="data_interpretation",
    name="Data interpretation",
    description="Tables, metrics, and trend reasoning.",
    version="1.0.0",
    priority=68,
    max_prompt_chars=360,
    activation_triggers=(
        "csv",
        "chart",
        "trend",
        "metric",
        "percentage",
        "dataset",
        "statistics",
        "correlation",
    ),
    prompt_fragment=(
        "Interpret data methodically: define units → describe pattern → note limitations → "
        "avoid overclaiming from small samples."
    ),
)

CREATIVE_WRITING = BuiltinSkill(
    id="creative_writing",
    name="Creative writing",
    description="Fiction and poetry constraints with imaginative scaffolding.",
    version="1.0.0",
    priority=55,
    max_prompt_chars=360,
    mutual_exclusion_group="technical_creative",
    activation_triggers=(
        "story",
        "poem",
        "character",
        "fiction",
        "narrative arc",
        "plot twist",
        "verse",
    ),
    activation_patterns=(re.compile(r"\bwrite (a|an) (story|poem|scene)\b", re.I),),
    prompt_fragment=(
        "Honor genre and tone requests. Establish setting and voice early; "
        "keep internal consistency; end with a satisfying beat unless asked to continue."
    ),
)
