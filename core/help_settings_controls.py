"""Extract settings control labels from section source for help corpus generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from core.diagnostic_logs import iter_diagnostic_logs
from core.ui_language import UI_LANGUAGE_LABELS
from ui.views.settings.registry import SETTINGS_SECTIONS, get_section

_SECTIONS_DIR = Path(__file__).resolve().parent.parent / "ui" / "views" / "settings" / "sections"
_HANDLERS_DIR = Path(__file__).resolve().parent.parent / "ui" / "views" / "settings" / "handlers"

SECTION_SOURCE_FILES: dict[str, tuple[str, ...]] = {
    "voice.audio": ("voice_audio.py",),
    "ai.models": ("ai_models.py",),
    "memory": ("memory.py",),
    "knowledge": (
        "knowledge.py",
        "knowledge_sources.py",
        "knowledge_web_discovery.py",
        "knowledge_presets.py",
        "knowledge_custom_sources.py",
        "knowledge_diagnostics.py",
        "knowledge_provider_status.py",
    ),
    "general": ("general.py",),
    "appearance.themes": ("appearance_themes.py",),
    "companion.desktop": ("desktop_companion.py",),
    "notifications": ("notifications.py",),
    "help": ("help.py",),
    "contact.feedback": ("contact_feedback.py",),
    "advanced": ("advanced.py",),
}

SECTION_SLUGS: dict[str, str] = {
    "voice.audio": "voice-audio",
    "ai.models": "ai-models",
    "memory": "memory",
    "knowledge": "knowledge",
    "general": "general",
    "appearance.themes": "themes",
    "companion.desktop": "desktop-companion",
    "notifications": "notifications",
    "help": "help",
    "contact.feedback": "contact-feedback",
    "advanced": "advanced",
}

_SECTIONS_WITH_RESET_FOOTER = frozenset(
    {
        "voice.audio",
        "ai.models",
        "memory",
        "knowledge",
        "general",
        "appearance.themes",
        "companion.desktop",
        "notifications",
    }
)

_KNOWLEDGE_LIBRARY_SEARCH_LABELS = (
    "Enable Local Knowledge Base",
    "Enable NLP Auto-Activator",
)

_KNOWLEDGE_LIBRARY_PRO_DEPTH_LABELS = (
    "Default precision ingest on import",
    "Precision retrieval",
)

_HELP_UNINSTALL_KEEP_VARIANTS = frozenset(
    {
        "Remove Qube package only…",
        "Remove Qube package only… (Linux)",
        "Remove Qube app only…",
        "Remove Qube app only… (macOS)",
    }
)

_HELP_UNINSTALL_STABLE_LABELS: tuple[str, ...] = (
    "Remove Qube package only… (Linux)",
    "Remove Qube app only… (macOS)",
)

_KNOWLEDGE_WEB_DISCOVERY_LABELS = frozenset(
    {
        "Privacy tier",
        "Live DDG usage",
        "Session limit override",
        "SearXNG base URL",
        "Slow down live DuckDuckGo searches slightly (recommended)",
        "Show advanced discovery limits",
        "Reset discovery health",
    }
)

_SKIP_FIELD_LABELS = frozenset({""})
_SKIP_CHECKBOX_LABELS = frozenset({"Reset to default configuration"})
_SKIP_CONTROL_LABELS = frozenset(
    {
        "Edit settings.json",
        "Open logs folder",
    }
)

def _should_skip_control_label(text: str) -> bool:
    lowered = text.lower()
    if "is not installed" in lowered or "are not ready" in lowered:
        return True
    return False

_SUBSECTION_RE = re.compile(
    r'add_subsection_to_(?:form|layout)\(\s*[^,]+,\s*(?:tr\()?["\']([^"\']+)["\']',
)
_ADD_ROW_RE = re.compile(r'\.addRow\(\s*["\']([^"\']+)["\']')
_CHECKBOX_RE = re.compile(
    r'QCheckBox\(\s*(?:tr\()?["\']([^"\']+)["\']',
    re.MULTILINE,
)
_LABEL_TEXT_RE = re.compile(
    r'label_text\s*=\s*(?:\(\s*)?["\']((?:[^"\']|\n)+?)["\'](?:\s*\))?',
    re.MULTILINE,
)
_DISCLOSURE_RE = re.compile(
    r'make_disclosure_row\(\s*[^,]+,\s*(?:tr\()?["\']([^"\']+)["\']',
)
_PRESTIGE_LABEL_RE = re.compile(
    r'\w+_label\s*=\s*QLabel\(\s*\n?\s*["\']([^"\']+)["\']',
    re.MULTILINE,
)
_QPUSHBUTTON_RE = re.compile(
    r'QPushButton\(\s*(?:tr\()?["\']([^"\']{2,80})["\']',
)
_GENERATION_ROW_RE = re.compile(
    r'_add_generation_form_row\([^,]+,\s*["\']([^"\']+)["\']',
)
_FORM_COMPANION_LABEL_RE = re.compile(
    r'QLabel\(\s*["\']([^"\']{3,60})["\']',
)
_BOOTSTRAP_BUTTON_RE = re.compile(
    r'button_text\s*=\s*["\']([^"\']+)["\']',
)


@dataclass
class _ControlBlock:
    subsection: str | None = None
    items: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SettingsControlEntry:
    subsection: str
    label: str
    kind: str


def settings_section_slug(section_id: str) -> str:
    slug = SECTION_SLUGS.get(section_id)
    if slug is None:
        raise KeyError(f"unknown settings section: {section_id}")
    return slug


def _read_section_sources(section_id: str) -> str:
    parts: list[str] = []
    for name in SECTION_SOURCE_FILES.get(section_id, ()):
        path = _SECTIONS_DIR / name
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
    if not parts:
        raise FileNotFoundError(f"no source files for settings section {section_id}")
    return "\n".join(parts)


def _read_handler_snippet(handler_name: str) -> str:
    path = _HANDLERS_DIR / handler_name
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _normalize_bootstrap_label(text: str) -> str | None:
    cleaned = " ".join(text.split())
    if not cleaned:
        return None
    if len(cleaned) > 90:
        cleaned = cleaned[:87].rstrip() + "…"
    return cleaned


def _append_unique(blocks: list[_ControlBlock], subsection: str, label: str) -> None:
    text = label.strip()
    if (
        not text
        or text in _SKIP_FIELD_LABELS
        or text in _SKIP_CHECKBOX_LABELS
        or text in _SKIP_CONTROL_LABELS
        or _should_skip_control_label(text)
    ):
        return
    if not blocks:
        blocks.append(_ControlBlock(subsection=subsection, items=[]))
    block = blocks[-1]
    if block.subsection != subsection:
        blocks.append(_ControlBlock(subsection=subsection, items=[]))
        block = blocks[-1]
    if text not in block.items:
        block.items.append(text)


def _scan_source_text(text: str, blocks: list[_ControlBlock]) -> None:
    current_subsection = "General"
    events: list[tuple[int, str, str]] = []

    for match in _SUBSECTION_RE.finditer(text):
        events.append((match.start(), "subsection", match.group(1).strip()))

    for match in _ADD_ROW_RE.finditer(text):
        events.append((match.start(), "field", match.group(1).strip()))

    for match in _CHECKBOX_RE.finditer(text):
        events.append((match.start(), "checkbox", match.group(1).strip()))

    for match in _LABEL_TEXT_RE.finditer(text):
        raw = match.group(1).strip()
        normalized = _normalize_bootstrap_label(raw)
        if normalized:
            events.append((match.start(), "checkbox", normalized))

    for match in _DISCLOSURE_RE.finditer(text):
        events.append((match.start(), "checkbox", match.group(1).strip()))

    for match in _PRESTIGE_LABEL_RE.finditer(text):
        events.append((match.start(), "checkbox", match.group(1).strip()))

    for match in _GENERATION_ROW_RE.finditer(text):
        events.append((match.start(), "field", match.group(1).strip()))

    for match in _BOOTSTRAP_BUTTON_RE.finditer(text):
        events.append((match.start(), "action", match.group(1).strip()))

    for match in _QPUSHBUTTON_RE.finditer(text):
        label = match.group(1).strip()
        if label.startswith("View ") or label.startswith("Clear "):
            events.append((match.start(), "action", label))
        else:
            events.append((match.start(), "action", label))

    for match in _FORM_COMPANION_LABEL_RE.finditer(text):
        label = match.group(1).strip()
        if label.startswith("Edit ") or label.startswith("Open "):
            continue
        if label in {"Promotion preset", "Default units"}:
            events.append((match.start(), "field", label))

    events.sort(key=lambda item: item[0])
    for _, kind, label in events:
        if kind == "subsection":
            current_subsection = label
            if not blocks or blocks[-1].items:
                blocks.append(_ControlBlock(subsection=current_subsection, items=[]))
            else:
                blocks[-1].subsection = current_subsection
            continue
        _append_unique(blocks, current_subsection, label)


def _inject_knowledge_library_pro_depth(blocks: list[_ControlBlock]) -> None:
    target = _ControlBlock(subsection="Library Pro depth", items=[])
    for label in _KNOWLEDGE_LIBRARY_PRO_DEPTH_LABELS:
        if label not in target.items:
            target.items.append(label)
    insert_at = len(blocks)
    for idx, block in enumerate(blocks):
        if block.subsection == "Library Pro depth":
            blocks[idx] = target
            return
        if block.subsection == "Retrieval profile":
            insert_at = idx
            break
    blocks.insert(insert_at, target)


def _inject_knowledge_library_search_phrases(blocks: list[_ControlBlock]) -> None:
    target = _ControlBlock(subsection="Library search phrases", items=[])
    for label in _KNOWLEDGE_LIBRARY_SEARCH_LABELS:
        if label not in target.items:
            target.items.append(label)
    insert_at = 0
    for idx, block in enumerate(blocks):
        if block.subsection == "Library search phrases":
            insert_at = idx
            blocks[insert_at] = target
            return
        if block.subsection == "Search quality":
            insert_at = idx
            break
    blocks.insert(insert_at, target)


def _stabilize_help_uninstall_labels(blocks: list[_ControlBlock]) -> None:
    """Emit both platform uninstall labels — source uses runtime sys.platform."""
    for block in blocks:
        if block.subsection != "Uninstall Qube":
            continue
        block.items = [
            item for item in block.items if item not in _HELP_UNINSTALL_KEEP_VARIANTS
        ]
        for label in _HELP_UNINSTALL_STABLE_LABELS:
            if label not in block.items:
                block.items.append(label)
        return


def _reassign_knowledge_web_discovery(blocks: list[_ControlBlock]) -> None:
    web_items: list[str] = []
    for block in blocks:
        kept: list[str] = []
        for item in block.items:
            if item in _KNOWLEDGE_WEB_DISCOVERY_LABELS:
                if item not in web_items:
                    web_items.append(item)
                continue
            kept.append(item)
        block.items = kept

    if not web_items:
        return

    for idx, block in enumerate(blocks):
        if block.subsection == "Web search discovery":
            for item in web_items:
                _append_unique(blocks, block.subsection, item)
            return

    blocks.append(_ControlBlock(subsection="Web search discovery", items=web_items))


def extract_settings_controls(section_id: str) -> list[SettingsControlEntry]:
    """Return ordered control labels for a settings section."""
    text = _read_section_sources(section_id)
    blocks: list[_ControlBlock] = []

    if section_id == "general":
        blocks.append(_ControlBlock(subsection="Language", items=[]))
        for label in UI_LANGUAGE_LABELS.values():
            _append_unique(blocks, "Language", label)

    _scan_source_text(text, blocks)

    if section_id == "knowledge":
        trigger_text = _read_handler_snippet("memory.py")
        for match in _CHECKBOX_RE.finditer(trigger_text):
            label = match.group(1).strip()
            if label in _KNOWLEDGE_LIBRARY_SEARCH_LABELS:
                continue
        _inject_knowledge_library_search_phrases(blocks)
        _inject_knowledge_library_pro_depth(blocks)
        _reassign_knowledge_web_discovery(blocks)

    if section_id == "help":
        _stabilize_help_uninstall_labels(blocks)

    if section_id == "advanced":
        for spec in iter_diagnostic_logs():
            blocks.append(_ControlBlock(subsection=spec.title, items=[]))
            if spec.supports_recording_toggle:
                toggle_label = spec.recording_toggle_label or "Record entries to this log"
                _append_unique(blocks, spec.title, toggle_label)
            _append_unique(blocks, spec.title, f"View {spec.title}")
            _append_unique(blocks, spec.title, "Clear log")

    entries: list[SettingsControlEntry] = []
    for block in blocks:
        subsection = block.subsection or "General"
        if not block.items and subsection:
            entries.append(SettingsControlEntry(subsection, "(section)", "group"))
            continue
        for item in block.items:
            entries.append(SettingsControlEntry(subsection, item, "control"))
    return entries


def generate_settings_controls_markdown(section_id: str) -> str:
    section = get_section(section_id)
    title = section.title if section else section_id
    lines = [
        "<!-- GENERATED CONTROLS — do not edit. Run: python scripts/generate_help_reference.py -->",
        f"Controls listed top-to-bottom for **Settings → {title}**.",
        "",
    ]

    current_subsection: str | None = None
    for entry in extract_settings_controls(section_id):
        if entry.subsection != current_subsection:
            current_subsection = entry.subsection
            lines.extend(["", f"### {current_subsection}", ""])
        if entry.label == "(section)":
            continue
        lines.append(f"- **{entry.label}**")

    if section_id in _SECTIONS_WITH_RESET_FOOTER:
        lines.extend(
            [
                "",
                "- **Reset to default configuration** — restores all settings on this page",
                "",
            ]
        )
    else:
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def generate_all_settings_controls() -> dict[str, str]:
    """Map generated/controls/{slug}.md → markdown body."""
    out: dict[str, str] = {}
    for section in SETTINGS_SECTIONS:
        slug = settings_section_slug(section.id)
        out[f"controls/{slug}.md"] = generate_settings_controls_markdown(section.id)
    return out
