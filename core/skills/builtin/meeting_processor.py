"""Meeting processor skill."""

from __future__ import annotations

import re

from core.skills.base import BuiltinSkill

_TRANSCRIPT_CUE = re.compile(
    r"\b(meeting notes|action items?|from (the|our) (call|meeting)|"
    r"who owns|follow[- ]?ups?|minutes from)\b",
    re.I,
)

MEETING_PROCESSOR = BuiltinSkill(
    id="meeting_processor",
    name="Meeting processor",
    description="Extract decisions, owners, and open issues from conversation notes.",
    version="1.0.0",
    priority=78,
    max_prompt_chars=400,
    activation_triggers=(
        "meeting notes",
        "action items",
        "from our call",
        "from the meeting",
        "who owns",
        "follow-up",
        "follow up",
        "summarize this meeting",
        "meeting summary",
        "unresolved",
    ),
    activation_patterns=(_TRANSCRIPT_CUE,),
    prompt_fragment=(
        "Extract from the content: Decisions made → Action items (owner, deadline if stated) → "
        "Open questions / unresolved issues → Brief outcome summary. "
        "Use bullets; mark unknown owners as 'TBD'. Do not fabricate attendees or commitments."
    ),
)
