"""Shared discourse type aliases (no cross-module imports)."""
from __future__ import annotations

from typing import Literal

TopicType = Literal["entity", "game", "concept", "task", "city", "person", "org", "unknown"]
ReferentSource = Literal[
    "assistant_answer",
    "assistant_pattern",
    "prior_session",
    "user_question",
    "history_scan",
    "none",
]
