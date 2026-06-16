"""Settings section registry — stable IDs, titles, icons, sidebar groups."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SettingsSectionDef:
    id: str
    title: str
    icon: str
    legacy_titles: tuple[str, ...] = ()
    group: str | None = None
    svg_icon: tuple[str, ...] | None = None


SETTINGS_SECTIONS: tuple[SettingsSectionDef, ...] = (
    SettingsSectionDef(
        id="voice.audio",
        title="Voice & Audio",
        icon="fa5s.microphone",
        legacy_titles=("AUDIO & HARDWARE",),
        group="Voice & Input",
    ),
    SettingsSectionDef(
        id="ai.models",
        title="AI & Models",
        icon="fa5s.brain",
        legacy_titles=(
            "AI MODELS & ROUTING",
            "NATIVE ENGINE & LOCAL LIBRARY",
            "STARTUP BEHAVIOR",
            "CHAT",
        ),
        group="Intelligence",
        svg_icon=("assets", "icons", "ai.svg"),
    ),
    SettingsSectionDef(
        id="memory",
        title="Memory",
        icon="fa5s.memory",
        legacy_titles=("MEMORY & PERFORMANCE", "Memory & Knowledge"),
        group="Intelligence",
    ),
    SettingsSectionDef(
        id="knowledge",
        title="Knowledge",
        icon="fa5s.book",
        legacy_titles=("NLP RAG TRIGGERS",),
        group="Intelligence",
    ),
    SettingsSectionDef(
        id="companion.desktop",
        title="Desktop Companion",
        icon="fa5s.ghost",
        legacy_titles=("DESKTOP COMPANION",),
        group="Interface",
    ),
    SettingsSectionDef(
        id="notifications",
        title="Notifications",
        icon="fa5s.bell",
        legacy_titles=("NOTIFICATIONS",),
        group="Interface",
    ),
    SettingsSectionDef(
        id="help",
        title="Help",
        icon="fa5s.question-circle",
        legacy_titles=("HELP & GUIDANCE",),
        group="Support",
    ),
    SettingsSectionDef(
        id="contact.feedback",
        title="Contact & Feedback",
        icon="fa5s.envelope",
        legacy_titles=("CONTACT & FEEDBACK",),
        group="Support",
    ),
    SettingsSectionDef(
        id="advanced",
        title="Advanced",
        icon="fa5s.cog",
        legacy_titles=("JSON SETTINGS",),
        group="System",
    ),
)

_SECTION_BY_ID: dict[str, SettingsSectionDef] = {s.id: s for s in SETTINGS_SECTIONS}

_LEGACY_TITLE_TO_ID: dict[str, str] = {}
for _sec in SETTINGS_SECTIONS:
    for _legacy in _sec.legacy_titles:
        _LEGACY_TITLE_TO_ID[_legacy] = _sec.id
    _LEGACY_TITLE_TO_ID[_sec.title] = _sec.id


def resolve_section_id(section: str) -> str | None:
    """Map section id, display title, or legacy title to a stable section id."""
    if section in _SECTION_BY_ID:
        return section
    if section in _LEGACY_TITLE_TO_ID:
        return _LEGACY_TITLE_TO_ID[section]
    return None


def get_section(section_id: str) -> SettingsSectionDef | None:
    return _SECTION_BY_ID.get(section_id)
