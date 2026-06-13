"""Software engineering skill."""

from __future__ import annotations

import re

from core.skills.base import BuiltinSkill

_CODE_FENCE = re.compile(r"```", re.I)
_FILE_EXT = re.compile(
    r"\.(py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|rb|php|sql|yaml|yml|toml|json)\b",
    re.I,
)

SOFTWARE_ENGINEERING = BuiltinSkill(
    id="software_engineering",
    name="Software engineering",
    description="Code reasoning scaffold: requirements, constraints, minimal solution.",
    version="1.0.0",
    priority=85,
    max_prompt_chars=420,
    mutual_exclusion_group="technical_creative",
    activation_triggers=(
        "bug",
        "refactor",
        "implement",
        "function",
        "api",
        "test",
        "debug",
        "compile",
        "error",
        "stack trace",
        "pull request",
        "codebase",
    ),
    activation_patterns=(_CODE_FENCE, _FILE_EXT),
    prompt_fragment=(
        "For technical tasks: clarify requirements → identify constraints → "
        "propose minimal solution → note edge cases/tests. "
        "Prefer runnable snippets over pseudocode when the user asks for code."
    ),
)
