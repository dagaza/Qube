"""Composer @-mention discoverability — hints and recent token persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.composer_attachments import ComposerAttachment, composer_tool_by_id
from core.composer_skills import ComposerSkillMention, skill_mention_from_id
from core.paths import user_data_root

RecentMentionKind = Literal["tool", "skill", "file", "conversation"]

RECENT_TOKENS_FILENAME = "composer_recent_tokens.json"
MAX_RECENT_MENTION_TOKENS = 8

COMPOSER_IDLE_PLACEHOLDER = "Message Qube… type @ for tools, files, and skills"

NEW_CHAT_TRANSCRIPT_HINT = (
    "New chat. Type a message or say your wake word.\n\n"
    "Tip: type @ to attach tools and files — try @library, @internet, or @research."
)

EMPTY_SESSION_TRANSCRIPT_HINT = (
    "No messages yet. Type @ in the composer to browse tools, files, skills, and help."
)

DEFAULT_SUGGESTION_TOOL_IDS: tuple[str, ...] = (
    "library",
    "internet",
    "help",
    "research",
)


@dataclass(frozen=True)
class RecentMention:
    kind: RecentMentionKind
    id: str
    label: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> RecentMention | None:
        if not isinstance(raw, dict):
            return None
        kind = str(raw.get("kind") or "").strip().lower()
        mention_id = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or "").strip()
        if kind not in ("tool", "skill", "file", "conversation") or not mention_id:
            return None
        if not label:
            label = mention_id
        return cls(kind=kind, id=mention_id, label=label)

    def to_mapping(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id, "label": self.label}

    def storage_key(self) -> str:
        return f"{self.kind}:{self.id}"


def recent_tokens_path() -> Path:
    return user_data_root() / RECENT_TOKENS_FILENAME


def _load_recent_raw() -> list[dict[str, Any]]:
    path = recent_tokens_path()
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    tokens = payload.get("tokens") if isinstance(payload, dict) else payload
    if not isinstance(tokens, list):
        return []
    return [item for item in tokens if isinstance(item, dict)]


def _save_recent_raw(entries: list[dict[str, Any]]) -> None:
    path = recent_tokens_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"tokens": entries}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def record_recent_mention(
    *,
    kind: RecentMentionKind,
    mention_id: str,
    label: str,
) -> None:
    """Persist a recently used @ mention (most recent first, deduped)."""
    cleaned_id = str(mention_id or "").strip()
    cleaned_label = str(label or cleaned_id).strip() or cleaned_id
    if not cleaned_id:
        return
    mention = RecentMention(kind=kind, id=cleaned_id, label=cleaned_label)
    existing = [
        parsed
        for parsed in (RecentMention.from_mapping(item) for item in _load_recent_raw())
        if parsed is not None
    ]
    filtered = [item for item in existing if item.storage_key() != mention.storage_key()]
    updated = [mention, *filtered][:MAX_RECENT_MENTION_TOKENS]
    _save_recent_raw([item.to_mapping() for item in updated])


def record_recent_attachment(attachment: ComposerAttachment) -> None:
    record_recent_mention(kind=attachment.kind, mention_id=attachment.id, label=attachment.label)


def record_recent_skill(mention: ComposerSkillMention) -> None:
    record_recent_mention(kind="skill", mention_id=mention.id, label=mention.label)


def list_recent_mentions(*, limit: int = MAX_RECENT_MENTION_TOKENS) -> list[RecentMention]:
    mentions: list[RecentMention] = []
    for raw in _load_recent_raw():
        parsed = RecentMention.from_mapping(raw)
        if parsed is not None:
            mentions.append(parsed)
        if len(mentions) >= max(1, limit):
            break
    return mentions


def default_suggestion_mentions() -> list[RecentMention]:
    suggestions: list[RecentMention] = []
    for tool_id in DEFAULT_SUGGESTION_TOOL_IDS:
        tool = composer_tool_by_id(tool_id)
        if tool is None:
            continue
        suggestions.append(
            RecentMention(
                kind="tool",
                id=str(tool["id"]),
                label=str(tool["label"]),
            )
        )
    return suggestions


def composer_hint_entries(*, limit: int = 6) -> tuple[list[RecentMention], bool]:
    """Return mention chips for the composer row and whether they are defaults."""
    recents = list_recent_mentions(limit=limit)
    if recents:
        return recents[:limit], False
    return default_suggestion_mentions()[:limit], True


def resolve_recent_mention(mention: RecentMention) -> ComposerAttachment | ComposerSkillMention | None:
    if mention.kind == "skill":
        resolved = skill_mention_from_id(mention.id)
        return resolved or ComposerSkillMention(id=mention.id, label=mention.label)
    if mention.kind == "tool":
        tool = composer_tool_by_id(mention.id)
        label = str(tool["label"]) if tool else mention.label
        return ComposerAttachment(kind="tool", id=mention.id, label=label)
    if mention.kind == "file":
        return ComposerAttachment(kind="file", id=mention.id, label=mention.label)
    if mention.kind == "conversation":
        return ComposerAttachment(kind="conversation", id=mention.id, label=mention.label)
    return None
