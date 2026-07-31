"""Unified @-mention search: browse, global search, and scoped browse helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.composer_attachments import ComposerAttachment, composer_tools_for_palette
from core.composer_commands import COMPOSER_COMMANDS, ComposerCommand
from core.composer_mention_trigger import filter_root_row_indices
from core.composer_skills import ComposerSkillMention, list_skill_mentions_for_palette

_MENTION_ROOT_CATEGORIES: tuple[tuple[str, str, str], ...] = (
    ("file", "Files", "Reference a library document"),
    ("conversation", "Conversations", "Reference another chat"),
    ("tool", "Tools", "Internet, library, or memory"),
    ("skill", "Skills", "Reasoning frameworks"),
    ("command", "Commands", "App actions and guidance"),
)

_SECTION_ORDER = (
    "categories",
    "tools",
    "integrations",
    "files",
    "conversations",
    "skills",
    "commands",
)

_SECTION_LABELS = {
    "categories": "Categories",
    "tools": "Tools",
    "integrations": "Integrations",
    "files": "Files",
    "conversations": "Conversations",
    "skills": "Skills",
    "commands": "Commands",
}

# Curated aliases → tool id (substring / prefix boost in search).
_TOOL_ALIASES: dict[str, str] = {
    "web": "internet",
    "inter": "internet",
    "net": "internet",
    "lib": "library",
    "mem": "memory",
    "scientific": "evidence",
}


class ComposerPaletteView(str, Enum):
    BROWSE = "browse"
    SEARCH = "search"
    SCOPED = "scoped"


@dataclass(frozen=True)
class MentionSearchHit:
    section: str
    score: int
    label: str
    subtitle: str
    payload: Any


def resolve_palette_view(query: str, *, scoped_kind: str | None) -> ComposerPaletteView:
    q = (query or "").strip()
    if scoped_kind:
        return ComposerPaletteView.SCOPED
    if q:
        return ComposerPaletteView.SEARCH
    return ComposerPaletteView.BROWSE


def _category_meta(kind: str) -> tuple[str, str, str]:
    for k, title, subtitle in _MENTION_ROOT_CATEGORIES:
        if k == kind:
            return title, subtitle, kind
    return kind.title(), "", kind


def _category_consumed_tokens(kind: str) -> set[str]:
    title, _subtitle, kind_l = _category_meta(kind)
    title_l = title.lower()
    tokens = {kind_l, title_l}
    if title_l.endswith("s"):
        tokens.add(title_l[:-1])
    return tokens


def resolve_scoped_filter(kind: str, raw_query: str) -> str:
    """In-section filter after stripping consumed category routing tokens."""
    q = (raw_query or "").strip().lower()
    if not q:
        return ""
    if q in _category_consumed_tokens(kind):
        return ""
    return q


def _score_text(q: str, *, text: str, exact: bool = False) -> int:
    t = text.lower()
    if not q:
        return 0
    if exact and q == t:
        return 100
    if t.startswith(q):
        return 80
    if q in t:
        return 50
    return 0


def _score_tool(q: str, tool: dict[str, str]) -> int:
    label = tool["label"]
    desc = tool["description"]
    tool_id = tool["id"]
    best = max(
        _score_text(q, text=tool_id, exact=True),
        _score_text(q, text=label),
        _score_text(q, text=desc),
    )
    alias_target = _TOOL_ALIASES.get(q)
    if alias_target and alias_target == tool_id:
        best = max(best, 90)
    for alias, target in _TOOL_ALIASES.items():
        if target == tool_id and (q.startswith(alias) or alias.startswith(q)):
            best = max(best, 75)
    return best


def _score_command(q: str, command: ComposerCommand) -> int:
    return max(
        _score_text(q, text=command.id, exact=True),
        _score_text(q, text=command.label),
        _score_text(q, text=command.description),
    )


def _score_skill(q: str, mention: ComposerSkillMention, description: str = "") -> int:
    return max(
        _score_text(q, text=mention.id, exact=True),
        _score_text(q, text=mention.label),
        _score_text(q, text=description),
    )


def _indexed_document(doc: dict, *, store) -> bool:
    filename = str(doc.get("filename") or "").strip()
    if not filename:
        return False
    chunk_count = int(doc.get("chunk_count") or 0)
    indexed = chunk_count > 0
    if store is not None:
        try:
            indexed = indexed or store.source_exists(filename)
        except Exception:
            pass
    return indexed


def search_composer_mentions(
    query: str,
    *,
    db,
    store=None,
    active_session_id: str | None = None,
    file_limit: int = 12,
    conversation_limit: int = 12,
    skill_limit: int = 8,
    command_limit: int = 8,
    integration_limit: int = 12,
) -> list[MentionSearchHit]:
    """Global grouped search across categories, tools, integrations, files, chats, skills, commands."""
    q = (query or "").strip().lower()
    if not q:
        return []

    hits: list[MentionSearchHit] = []

    for idx in filter_root_row_indices(q):
        kind, title, subtitle = _MENTION_ROOT_CATEGORIES[idx]
        score = max(
            _score_text(q, text=title),
            _score_text(q, text=kind),
            _score_text(q, text=subtitle),
        )
        if score > 0:
            hits.append(
                MentionSearchHit(
                    section="categories",
                    score=score,
                    label=title,
                    subtitle=subtitle,
                    payload=("category", kind),
                )
            )

    for tool in composer_tools_for_palette(q):
        score = _score_tool(q, tool)
        if score <= 0:
            continue
        label = tool["label"]
        desc = tool["description"]
        hits.append(
            MentionSearchHit(
                section="tools",
                score=score,
                label=label,
                subtitle=desc,
                payload=ComposerAttachment(kind="tool", id=tool["id"], label=label),
            )
        )

    from core.integrations.search import search_integrations_capabilities

    integration_count = 0
    for entry in search_integrations_capabilities(q, limit=integration_limit * 2):
        if entry.score <= 0:
            continue
        prefix = "[lock] " if entry.locked else ""
        hits.append(
            MentionSearchHit(
                section="integrations",
                score=entry.score,
                label=f"{prefix}{entry.label}",
                subtitle=entry.subtitle,
                payload=entry,
            )
        )
        integration_count += 1
        if integration_count >= integration_limit:
            break

    if db is not None:
        try:
            docs = db.get_library_documents_for_sidebar_search(q, limit=file_limit * 4)
        except Exception:
            docs = []
        file_count = 0
        for doc in docs:
            if not _indexed_document(doc, store=store):
                continue
            filename = str(doc.get("filename") or "").strip()
            score = _score_text(q, text=filename)
            if score <= 0:
                continue
            hits.append(
                MentionSearchHit(
                    section="files",
                    score=score,
                    label=filename,
                    subtitle="Library document",
                    payload=ComposerAttachment(kind="file", id=filename, label=filename),
                )
            )
            file_count += 1
            if file_count >= file_limit:
                break

        try:
            sessions = db.get_sessions_for_sidebar_search(q, limit=conversation_limit * 2)
        except Exception:
            sessions = []
        conv_count = 0
        for sess in sessions:
            sid = str(sess.get("id") or "")
            if not sid or sid == active_session_id:
                continue
            title = str(sess.get("title") or "Untitled").strip()
            score = _score_text(q, text=title)
            if score <= 0:
                continue
            hits.append(
                MentionSearchHit(
                    section="conversations",
                    score=score,
                    label=title,
                    subtitle="Conversation",
                    payload=ComposerAttachment(
                        kind="conversation", id=sid, label=title
                    ),
                )
            )
            conv_count += 1
            if conv_count >= conversation_limit:
                break

    from core.skills.registry import get_skill

    skill_rows = list_skill_mentions_for_palette(query=q, limit=skill_limit * 2)
    skill_count = 0
    for mention in skill_rows:
        skill = get_skill(mention.id)
        desc = skill.description if skill is not None else ""
        score = _score_skill(q, mention, desc)
        if score <= 0:
            continue
        hits.append(
            MentionSearchHit(
                section="skills",
                score=score,
                label=mention.label,
                subtitle=desc,
                payload=mention,
            )
        )
        skill_count += 1
        if skill_count >= skill_limit:
            break

    cmd_count = 0
    for command in COMPOSER_COMMANDS:
        score = _score_command(q, command)
        if score <= 0:
            continue
        hits.append(
            MentionSearchHit(
                section="commands",
                score=score,
                label=command.label,
                subtitle=command.description,
                payload=command,
            )
        )
        cmd_count += 1
        if cmd_count >= command_limit:
            break

    return group_search_hits(hits)


def group_search_hits(hits: list[MentionSearchHit]) -> list[MentionSearchHit]:
    """Sort by section order, then score descending within each section."""
    grouped: dict[str, list[MentionSearchHit]] = {s: [] for s in _SECTION_ORDER}
    for hit in hits:
        if hit.section in grouped:
            grouped[hit.section].append(hit)
    ordered: list[MentionSearchHit] = []
    for section in _SECTION_ORDER:
        rows = grouped[section]
        rows.sort(key=lambda h: (-h.score, h.label.lower()))
        ordered.extend(rows)
    return ordered


def section_label(section: str) -> str:
    return _SECTION_LABELS.get(section, section.title())
