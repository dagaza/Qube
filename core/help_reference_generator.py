"""Build generated help reference markdown from live application registries."""

from __future__ import annotations

from collections import defaultdict

from core.composer_attachments import (
    COMPOSER_TOOLS,
    CONVERSATION_REF_BUDGET,
    _TOOL_USAGE_HINTS,
)
from core.composer_command_defs import COMPOSER_COMMANDS
from core.knowledge.adapters.catalog import ADAPTER_CATALOG
from core.knowledge.types import (
    SERVICE_FINANCE_KNOWLEDGE,
    SERVICE_LEGAL_KNOWLEDGE,
    SERVICE_SCIENTIFIC_EVIDENCE,
)
from core.skills.registry import iter_skills
from ui.views.settings.registry import SETTINGS_SECTIONS

GENERATED_BANNER = (
    "<!-- GENERATED FILE — do not edit. Run: python scripts/generate_help_reference.py -->\n"
)

_SERVICE_LABELS: dict[str, str] = {
    SERVICE_SCIENTIFIC_EVIDENCE: "Scientific literature",
    SERVICE_FINANCE_KNOWLEDGE: "Finance",
    SERVICE_LEGAL_KNOWLEDGE: "Legal",
}


def _md_heading_block(title: str, body_lines: list[str]) -> str:
    lines = [f"## {title}", ""]
    lines.extend(body_lines)
    lines.append("")
    return "\n".join(lines)


def _advanced_tool_ids() -> list[str]:
    return [str(tool["id"]) for tool in COMPOSER_TOOLS if tool.get("advanced")]


def _skills_mutual_exclusion_groups() -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for skill in iter_skills():
        group = getattr(skill, "mutual_exclusion_group", None)
        if group:
            groups[str(group)].append((skill.id, skill.name))
    return groups


def generate_composer_attachments_markdown() -> str:
    lines = [
        GENERATED_BANNER.rstrip(),
        "# Composer attachments (@file, @chat, routing)",
        "",
        "## Common questions",
        "",
        "- What is the difference between `@[file:…]`, `@[chat:…]`, and `@[tool:…]`?",
        "- Which `@` attachment controls routing when I mix several?",
        "- How do I reference another conversation in chat?",
        "",
        "## What attachments are",
        "",
        "The `@` palette has five categories: **Files**, **Conversations**, **Tools**, "
        "**Skills**, and **Commands**. This page covers the three **routing** attachment kinds "
        "that steer retrieval and context for a turn.",
        "",
        "Type **`@`** in the composer to browse categories, or keep typing to search everything "
        "at once. Pick **1–5** on the category menu for a shortcut; **Enter** or **Tab** selects.",
        "",
        "## Files — `@[file:filename.pdf]`",
        "",
        "Reference an indexed **Library** document. Forces search scoped to that file only "
        "(not your whole Library). Filenames containing **`]`** cannot be inserted from the palette.",
        "",
        "Configure Library search in **Settings → Knowledge** (master RAG switch and search models).",
        "",
        "## Conversations — `@[chat:session-id::Title]`",
        "",
        "Pull another chat's transcript into this turn. The referenced history replaces this "
        f"conversation's history for that turn only (about **{CONVERSATION_REF_BUDGET:,}** characters). "
        "It does not merge unrelated turns from the current chat.",
        "",
        "## Tools — `@[tool:…]`",
        "",
        "Route the turn to web discovery, Live Sources, Library search, Memory, Help, deep research, "
        "or a custom preset. See [Composer tools](composer-tools.md) for every built-in token.",
        "",
        "## Routing rule (order matters)",
        "",
        "You can insert several routing tokens, but only the **first** one in your message "
        "(left-to-right) controls behaviour — the first among `@[file:…]`, `@[chat:…]`, or "
        "`@[tool:…]`. Put the attachment you want **first**.",
        "",
        "Example: `@[tool:internet] @[file:doc.pdf]` uses web discovery, not the file.",
        "",
        "**Skills** (`@[skill:…]`) and **commands** do not participate in this rule. Skills add "
        "prompt guidance; commands run immediately and are not sent to the model.",
        "",
        "## Mixing skills with routing attachments",
        "",
        "Skills pair well with one routing attachment. Example:",
        "",
        "`@[skill:research_synthesis] @[tool:library] Summarize my uploaded notes.`",
        "",
        "## When attachments are skipped",
        "",
        "- Explicit **“remember …”** turns skip all attachments and all skills (including forced).",
        "- Unknown `@[tool:…]` or `@[skill:…]` IDs are ignored (logged); other tokens may still apply.",
        "",
        "## Also called",
        "",
        "composer routing, file attachments, conversation references, @ mentions",
        "",
        "## Related",
        "",
        "- [Composer tools](composer-tools.md) — built-in and preset `@[tool:…]` tokens",
        "- [Composer skills](composer-skills.md) — `@[skill:…]` reasoning frameworks",
        "- [Composer commands](composer-commands.md) — immediate palette actions",
        "- [What do @ mentions do FAQ](../faq/what-do-at-mentions-do.md)",
        "- [Chat with a library document](../workflows/chat-with-a-library-document.md)",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def generate_composer_tools_markdown() -> str:
    advanced_ids = _advanced_tool_ids()
    advanced_list = ", ".join(f"`{tool_id}`" for tool_id in advanced_ids)
    lines = [
        GENERATED_BANNER.rstrip(),
        "# Composer tools (@tool)",
        "",
        "## Common questions",
        "",
        "- What `@` tools can I attach in chat?",
        "- What does `@[tool:library]` do?",
        "- What is the difference between `@library` and `@internet`?",
        "- Why do some tools not appear in the `@` palette until I search?",
        "",
        "## What these tokens do",
        "",
        "Composer **tools** route a chat turn to a specific capability. Insert a token like "
        "`@[tool:library]` in your message, or pick one from the `@` palette under **Tools**.",
        "",
        "Only the **first routing attachment** in your message controls behaviour — the first "
        "token among `@[file:…]`, `@[chat:…]`, or `@[tool:…]` in left-to-right order. "
        "See [Composer attachments](composer-attachments.md) for files, chats, and mixing rules.",
        "",
        "## Built-in tools",
        "",
    ]

    for tool in COMPOSER_TOOLS:
        tool_id = str(tool["id"])
        label = str(tool["label"])
        desc = str(tool["description"])
        advanced = " (advanced palette)" if tool.get("advanced") else ""
        hint = _TOOL_USAGE_HINTS.get(tool_id, "")
        lines.append(f"### {label} — `@[tool:{tool_id}]`{advanced}")
        lines.append("")
        lines.append(desc)
        if hint:
            lines.append("")
            lines.append(hint)
        lines.append("")

    lines.extend(
        [
            "## Advanced palette tools",
            "",
            "These built-in tools are hidden from the default **Tools** browse list until you "
            f"type `@` and search for the tool id: {advanced_list}.",
            "",
            "Type the id (for example `pubmed`) or pick the token once it appears.",
            "",
            "## My knowledge presets (dynamic)",
            "",
            "Presets you create under **Settings → Knowledge → My knowledge** appear in the `@` "
            "palette as **`@[tool:user:…]`** tokens (for example `@[tool:user:biology]`). "
            "They bundle selected Live Source adapters or web-fetch domains — not Library folders.",
            "",
            "Preset tokens are not listed here because they depend on your configuration. "
            "See [Create a knowledge preset](../workflows/create-knowledge-preset.md).",
            "",
            "## Single-adapter pins (`@[tool:source:…]`)",
            "",
            "Pin one Live Source adapter manually with `@[tool:source:adapter_id]` "
            "(for example `@[tool:source:pubmed]`). These tokens are not shown in the palette; "
            "use a knowledge preset when you want a repeatable scoped bundle instead.",
            "",
            "## Settings and prerequisites",
            "",
            "- **Library** (`@[tool:library]`) — enable document search in **Settings → Knowledge** "
            "and prepare search models (see [Prepare search models for Library]"
            "(../workflows/prepare-search-models-for-library.md)).",
            "- **Web / evidence tools** (`@[tool:internet]`, `@[tool:evidence]`, `@[tool:trusted]`, etc.) "
            "— configure **Live Sources** and optional API keys in **Settings → Knowledge**. "
            "See [Live sources overview](live-sources-overview.md).",
            "- **Help** (`@[tool:help]`) — searches the **Qube Documentation** collection only. "
            "Help docs are excluded from normal Library search unless you attach `@help`.",
            "- **Memory** (`@[tool:memory]`) — requires Memory features enabled in "
            "**Settings → Memory**.",
            "",
            "## Also called",
            "",
            "composer attachments, @ mentions, tool tokens, routing tools",
            "",
            "## Related",
            "",
            "- [Composer attachments](composer-attachments.md) — files, chats, routing order",
            "- [Composer commands](composer-commands.md) — immediate app actions",
            "- [Composer skills](composer-skills.md) — reasoning frameworks (not routing)",
            "- [Knowledge settings](../features/settings/knowledge.md) — Live Sources and My knowledge",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def generate_composer_commands_markdown() -> str:
    from core.composer_command_defs import COMPOSER_COMMANDS

    lines = [
        GENERATED_BANNER.rstrip(),
        "# Composer commands",
        "",
        "## Common questions",
        "",
        "- What `@` commands run immediately without sending a prompt?",
        "- How do I reset Help & Guidance?",
        "",
        "## What commands are",
        "",
        "Composer **commands** run app actions when selected from the `@` palette. "
        "They are **not** sent to the language model.",
        "",
        "## Available commands",
        "",
    ]

    for cmd in COMPOSER_COMMANDS:
        lines.append(f"### {cmd.label}")
        lines.append("")
        lines.append(f"Palette id: `{cmd.id}`. {cmd.description}")
        if cmd.requires_confirmation:
            lines.append("")
            lines.append("Requires confirmation before running.")
        lines.append("")

    lines.extend(
        [
            "## Also called",
            "",
            "palette commands, app commands, immediate actions",
            "",
            "## Related",
            "",
            "- [Composer tools](composer-tools.md)",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def generate_composer_skills_markdown() -> str:
    exclusion_groups = _skills_mutual_exclusion_groups()
    lines = [
        GENERATED_BANNER.rstrip(),
        "# Composer skills (@skill)",
        "",
        "## Common questions",
        "",
        "- What are `@` skills?",
        "- How do I combine a skill with `@library` or `@internet`?",
        "- Can I attach more than one skill?",
        "",
        "## What skills are",
        "",
        "Composer **skills** add reasoning frameworks to the system prompt. "
        "They do **not** change routing — pair them with a tool or file attachment when you need retrieval.",
        "",
        "Example: `@[skill:research_synthesis] @[tool:library] Summarize my notes.`",
        "",
        "## Limits and mixing",
        "",
        "- Multiple `@[skill:…]` tokens are allowed; duplicates dedupe to one entry.",
        "- Up to **three** skills apply per turn by default (auto-detected plus forced combined).",
        "- Combined skill guidance respects a character budget (default **1200** characters).",
        "- Enable auto-detection in **Settings → AI & Models → Reasoning skills** "
        "(**Enable compositional reasoning skills**). Forced `@[skill:…]` tokens still work "
        "when that toggle is off, unless a skip condition below applies.",
        "",
        "## When skills are skipped",
        "",
        "- Explicit **“remember …”** turns skip all skills (including forced) and all attachments.",
        "- Unknown `@[skill:…]` IDs are ignored (logged); other skills may still run.",
        "",
        "## Mutual exclusion",
        "",
        "Only one skill per exclusion group can apply on a turn. If several qualify, the highest-scoring "
        "skill wins; forced `@[skill:…]` tokens take precedence over auto-detected peers in the same group.",
        "",
    ]

    if exclusion_groups:
        for group_name in sorted(exclusion_groups):
            members = exclusion_groups[group_name]
            member_text = ", ".join(
                f"**{name}** (`@[skill:{skill_id}]`)"
                for skill_id, name in sorted(members, key=lambda item: item[1].lower())
            )
            lines.append(f"- **{group_name}** — {member_text}")
        lines.append("")
    else:
        lines.extend(["(No mutual exclusion groups in the built-in registry.)", ""])

    lines.extend(
        [
            "## Built-in skills",
            "",
        ]
    )

    for skill in sorted(iter_skills(), key=lambda s: (s.name.lower(), s.id)):
        lines.append(f"### {skill.name} — `@[skill:{skill.id}]`")
        lines.append("")
        desc = (skill.description or "").strip()
        if desc:
            lines.append(desc)
        lines.append("")

    lines.extend(
        [
            "## Also called",
            "",
            "reasoning skills, skill tokens, prompt frameworks",
            "",
            "## Related",
            "",
            "- [Composer attachments](composer-attachments.md) — routing vs skills",
            "- [Composer tools](composer-tools.md)",
            "- [AI & Models settings](../features/settings/ai-models.md) — enable reasoning skills",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def generate_settings_sections_markdown() -> str:
    lines = [
        GENERATED_BANNER.rstrip(),
        "# Settings sections",
        "",
        "## Common questions",
        "",
        "- Where are Qube settings?",
        "- What settings sections exist?",
        "- How do I open AI & Models settings?",
        "",
        "## Where to find settings",
        "",
        "Open **Settings** from the main navigation, then choose a section in the left sidebar.",
        "",
        "## Section index",
        "",
        "| Section | Settings id | Sidebar group |",
        "|---------|-------------|---------------|",
    ]

    for section in SETTINGS_SECTIONS:
        group = section.group or ""
        lines.append(
            f"| {section.title} | `{section.id}` | {group} |"
        )

    lines.append("")
    current_group: str | None = None
    for section in SETTINGS_SECTIONS:
        if section.group and section.group != current_group:
            current_group = section.group
            lines.extend(["", f"## {current_group}", ""])
        lines.append(f"### {section.title} (`{section.id}`)")
        lines.append("")
        lines.append(f"Open **Settings → {section.title}**.")
        if section.legacy_titles:
            legacy = ", ".join(section.legacy_titles)
            lines.append("")
            lines.append(f"Also called: {legacy}")
        lines.append("")

    lines.extend(
        [
            "## Also called",
            "",
            "preferences, configuration, options, settings pages",
            "",
            "## Related",
            "",
            "- [Live sources overview](live-sources-overview.md) — Knowledge Live Sources adapters",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _adapter_key_flags(entry) -> list[str]:
    flags: list[str] = []
    if not entry.implemented:
        flags.append("planned")
    if entry.requires_api_key:
        flags.append("API key required")
    elif entry.optional_api_key:
        flags.append("optional API key")
    if not entry.default_enabled:
        flags.append("off by default")
    return flags


def generate_live_sources_overview_markdown() -> str:
    by_service: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seen_ids: dict[str, set[str]] = defaultdict(set)

    for entry in ADAPTER_CATALOG:
        if entry.id in seen_ids[entry.knowledge_service]:
            continue
        seen_ids[entry.knowledge_service].add(entry.id)
        by_service[entry.knowledge_service][entry.ui_group].append(entry)

    lines = [
        GENERATED_BANNER.rstrip(),
        "# Live sources overview",
        "",
        "## Common questions",
        "",
        "- What Live Sources are available in Knowledge settings?",
        "- Which scientific literature adapters does Qube support?",
        "- Do any sources require an API key?",
        "",
        "## What Live Sources are",
        "",
        "**Live Sources** (Settings → Knowledge) connect Qube to online catalogs — "
        "scientific papers, finance filings, legal opinions, and more. "
        "This page summarizes adapter metadata shipped with the app.",
        "",
        "Library document search and Live Sources are different: Library searches files you ingested; "
        "Live Sources query external services when `@internet`, `@evidence`, or related tools route to them.",
        "",
        "## Adapter catalog summary",
        "",
    ]

    for service_id in sorted(by_service):
        label = _SERVICE_LABELS.get(service_id, service_id.replace("_", " ").title())
        entries = by_service[service_id]
        implemented = sum(
            1
            for group_entries in entries.values()
            for item in group_entries
            if item.implemented
        )
        total = sum(len(group_entries) for group_entries in entries.values())
        lines.append(f"## {label} (`{service_id}`)")
        lines.append("")
        lines.append(
            f"{total} catalog entries ({implemented} implemented). "
            f"Configure defaults in **Settings → Knowledge → Live Sources**."
        )
        lines.append("")

        for ui_group in sorted(entries):
            group_entries = sorted(entries[ui_group], key=lambda e: e.label.lower())
            lines.append(f"### {ui_group}")
            lines.append("")
            for entry in group_entries:
                flags = _adapter_key_flags(entry)
                flag_text = f" — {', '.join(flags)}" if flags else ""
                lines.append(f"- **{entry.label}** (`{entry.id}`){flag_text}")
            lines.append("")

    lines.extend(
        [
            "## Also called",
            "",
            "internet search adapters, online lookup, external knowledge sources, evidence adapters",
            "",
            "## Related",
            "",
            "- [Knowledge settings](../features/settings/knowledge.md) — full Live Sources UI (Phase 3)",
            "- [Composer tools](composer-tools.md) — `@evidence`, `@finance`, `@legal`, and related tools",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


_REFERENCE_BUILDERS = {
    "reference/composer-attachments.md": generate_composer_attachments_markdown,
    "reference/composer-tools.md": generate_composer_tools_markdown,
    "reference/composer-commands.md": generate_composer_commands_markdown,
    "reference/composer-skills.md": generate_composer_skills_markdown,
    "reference/settings-sections.md": generate_settings_sections_markdown,
    "reference/live-sources-overview.md": generate_live_sources_overview_markdown,
}


def generate_all_reference_markdown() -> dict[str, str]:
    """Return relative paths under generated/reference/ → markdown content."""
    return {path: builder() for path, builder in _REFERENCE_BUILDERS.items()}
