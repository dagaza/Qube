"""Parse @help UI action blocks from assistant markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ACTION_LINE_RE = re.compile(
    r"^\[action:(?P<kind>[a-z_]+)"
    r"(?:\s+settings_section=(?P<section>[^\s\]]+))?"
    r'(?:\s+label="(?P<label>[^"]*)")?'
    r"\s*\]\s*$",
    re.MULTILINE,
)

_DEFAULT_LABELS: dict[str, str] = {
    "ai.models": "Open AI & Models settings",
    "knowledge": "Open Knowledge settings",
    "memory": "Open Memory settings",
    "voice.audio": "Open Voice & Audio settings",
    "general": "Open General settings",
    "notifications": "Open Notifications settings",
    "companion.desktop": "Open Desktop Companion settings",
    "help": "Open Help settings",
    "about": "Open About settings",
    "contact.feedback": "Open Contact & Feedback settings",
    "privacy.data": "Open Privacy & data settings",
    "diagnostics": "Open Diagnostics settings",
    "license": "Open License settings",
    "advanced": "Open Advanced settings",
}


@dataclass(frozen=True)
class HelpActionChip:
    kind: str
    settings_section: str
    label: str


def _default_label(settings_section: str) -> str:
    return _DEFAULT_LABELS.get(
        settings_section,
        f"Open {settings_section.replace('.', ' ')} settings",
    )


def parse_help_action_blocks(text: str) -> tuple[str, list[HelpActionChip]]:
    """Return markdown with action lines removed plus parsed chips."""
    actions: list[HelpActionChip] = []

    def _repl(match: re.Match[str]) -> str:
        kind = str(match.group("kind") or "").strip()
        if kind != "open_settings_section":
            return match.group(0)
        section = str(match.group("section") or "").strip()
        if not section:
            return match.group(0)
        label = str(match.group("label") or "").strip() or _default_label(section)
        actions.append(
            HelpActionChip(
                kind=kind,
                settings_section=section,
                label=label,
            )
        )
        return ""

    stripped = _ACTION_LINE_RE.sub(_repl, text or "")
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped, actions
