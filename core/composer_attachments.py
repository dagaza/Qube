"""Composer @-mention attachments: token format, parsing, and context builders."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from core.database import DatabaseManager

AttachmentKind = Literal["file", "conversation", "tool"]

_CHAT_TOKEN_SEP = "::"

_TOKEN_RE = re.compile(
    r"@\[(file|chat|tool):([^\]]+)\]",
    re.IGNORECASE,
)

CONVERSATION_REF_BUDGET = 7000

COMPOSER_TOOLS: list[dict[str, str | bool]] = [
    {
        "id": "trusted",
        "label": "Trusted",
        "description": "Wikipedia and allowlisted sources",
    },
    {
        "id": "evidence",
        "label": "Scientific literature",
        "description": "Peer-reviewed papers and preprints across disciplines",
    },
    {
        "id": "finance",
        "label": "Finance",
        "description": "SEC EDGAR company filings (10-K, 10-Q, 8-K)",
    },
    {
        "id": "legal",
        "label": "Legal",
        "description": "U.S. case law opinions via CourtListener",
    },
    {
        "id": "research",
        "label": "Deep research",
        "description": "Multi-step evidence report (async, non-blocking)",
    },
    {"id": "internet", "label": "Internet", "description": "Live web search"},
    {"id": "library", "label": "Library", "description": "Search your documents"},
    {"id": "memory", "label": "Memory", "description": "Search stored memories"},
    {
        "id": "science",
        "label": "Scientific literature",
        "description": "Alias for @evidence (same routing)",
        "advanced": True,
    },
    {
        "id": "wikipedia",
        "label": "Wikipedia",
        "description": "Wikipedia intro extracts only",
        "advanced": True,
    },
    {
        "id": "pubmed",
        "label": "PubMed",
        "description": "Biomedical literature abstracts",
        "advanced": True,
    },
    {
        "id": "arxiv",
        "label": "arXiv",
        "description": "Preprint abstracts (CS, physics, math)",
        "advanced": True,
    },
]

_WEB_COMPOSER_TOOLS = frozenset(
    {"internet", "trusted", "evidence", "science", "wikipedia", "pubmed", "arxiv", "finance", "legal"}
)

_TOOL_USAGE_HINTS: dict[str, str] = {
    "trusted": "Use for general facts from vetted allowlisted sources.",
    "evidence": "Use when you need cited papers; discipline-specific sources are chosen automatically.",
    "finance": "Use for SEC filings, company financials, and regulatory disclosures.",
    "legal": "Use for U.S. court opinions and case law.",
    "research": "Use for a multi-step async literature review report.",
    "internet": "Use for timely web information beyond your library.",
    "library": "Use to search only your uploaded documents.",
    "memory": "Use to recall facts saved from past chats.",
    "science": "Same routing as @evidence; prefer @evidence in the palette.",
    "wikipedia": "Use for quick encyclopedia summaries only.",
    "pubmed": "Use for biomedical papers and clinical research.",
    "arxiv": "Use for CS, physics, and math preprints.",
}


def _tool_matches_palette_filter(tool: dict[str, str | bool], query: str) -> bool:
    """Return whether a tool should appear in browse/search palettes for ``query``."""
    tool_id = str(tool["id"]).lower()
    if tool.get("advanced"):
        if not query:
            return False
        return query == tool_id or tool_id.startswith(query) or query.startswith(tool_id)
    if not query:
        return True
    label = str(tool["label"]).lower()
    desc = str(tool["description"]).lower()
    return query in label or query in desc or query in tool_id


def composer_tools_for_palette(query: str = "") -> list[dict[str, str | bool]]:
    """Tools shown in composer palettes; hides advanced aliases unless id matches."""
    q = (query or "").strip().lower()
    return [tool for tool in COMPOSER_TOOLS if _tool_matches_palette_filter(tool, q)]


def composer_tool_by_id(tool_id: str) -> dict[str, str | bool] | None:
    for tool in COMPOSER_TOOLS:
        if tool["id"] == tool_id:
            return tool
    return None


def composer_tool_tooltip(tool: dict[str, str | bool]) -> str:
    label = str(tool["label"])
    desc = str(tool["description"])
    hint = _TOOL_USAGE_HINTS.get(str(tool["id"]), "")
    parts = [f"{label}. {desc}"]
    if hint:
        parts.append(hint)
    return " ".join(parts) + f" Inserts @[tool:{tool['id']}]."

_ROLE_HEADINGS = {
    "user": "User",
    "assistant": "Assistant",
    "system": "System",
}


@dataclass(frozen=True)
class ComposerAttachment:
    kind: AttachmentKind
    id: str
    label: str


def _sanitize_conversation_label(label: str) -> str:
    """Keep labels readable inside ``@[chat:uuid::label]`` tokens."""
    cleaned = (label or "").strip().replace("]", "").replace(_CHAT_TOKEN_SEP, " ")
    return cleaned[:120] if cleaned else "Conversation"


def format_token(att: ComposerAttachment) -> str:
    """Return the plain-text token inserted into the composer."""
    if att.kind == "file":
        return f"@[file:{att.id}]"
    if att.kind == "conversation":
        title = _sanitize_conversation_label(att.label)
        return f"@[chat:{att.id}{_CHAT_TOKEN_SEP}{title}]"
    return f"@[tool:{att.id}]"


def validate_file_token(filename: str) -> bool:
    """Reject filenames that would break token parsing."""
    return bool(filename and "]" not in filename)


def parse_attachments(text: str) -> tuple[str, list[ComposerAttachment]]:
    """Split composer text into clean user prompt and structured attachments."""
    attachments: list[ComposerAttachment] = []
    seen: set[tuple[str, str]] = set()

    def _repl(match: re.Match[str]) -> str:
        kind_raw = match.group(1).lower()
        att_id = match.group(2).strip()
        if not att_id:
            return ""
        if kind_raw == "file":
            kind: AttachmentKind = "file"
            label = att_id
        elif kind_raw == "chat":
            kind = "conversation"
            if _CHAT_TOKEN_SEP in att_id:
                session_id, _, title = att_id.partition(_CHAT_TOKEN_SEP)
                att_id = session_id.strip()
                label = title.strip() or att_id[:8] + "…"
            else:
                label = att_id[:8] + "…" if len(att_id) > 12 else att_id
        else:
            kind = "tool"
            tool = next((t for t in COMPOSER_TOOLS if t["id"] == att_id), None)
            label = tool["label"] if tool else att_id
        key = (kind, att_id)
        if key not in seen:
            seen.add(key)
            attachments.append(ComposerAttachment(kind=kind, id=att_id, label=label))
        return ""

    clean = _TOKEN_RE.sub(_repl, text or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean, attachments


def strip_tokens_for_display(text: str) -> str:
    """Remove attachment tokens for compact display."""
    from core.composer_skills import strip_skill_tokens

    _enforced, without_skills = strip_skill_tokens(text or "")
    return re.sub(r"\s+", " ", _TOKEN_RE.sub("", without_skills)).strip()


def attachment_summary(attachments: list[ComposerAttachment]) -> str:
    if not attachments:
        return ""
    parts = [f"{a.kind}:{a.label}" for a in attachments]
    return ", ".join(parts)


def resolve_attachment_routing(
    attachments: list[ComposerAttachment],
) -> dict | None:
    """
    Return a routing decision patch from composer attachments, or None.
    First attachment by kind precedence: file > conversation > tool.
    """
    if not attachments:
        return None
    if len(attachments) > 1:
        import logging

        logging.getLogger("Qube.LLM").warning(
            "[LLM Worker] Multiple composer attachments; first drives routing: %s",
            attachment_summary(attachments),
        )
    primary = attachments[0]
    if primary.kind == "file":
        return {
            "route": "rag",
            "strategy": "attachment_file",
            "attachment_file": True,
            "rag_query": None,
            "source_filter": primary.id,
            "composer_attachments": _attachments_telemetry(attachments),
        }
    if primary.kind == "conversation":
        return {
            "route": "none",
            "strategy": "attachment_conversation",
            "attachment_conversation": True,
            "referenced_session_id": primary.id,
            "composer_attachments": _attachments_telemetry(attachments),
        }
    tool_id = primary.id
    if tool_id in _WEB_COMPOSER_TOOLS:
        return {
            "route": "web",
            "strategy": f"attachment_tool_{tool_id}",
            "attachment_tool": tool_id,
            "composer_attachments": _attachments_telemetry(attachments),
        }
    if tool_id == "research":
        return {
            "route": "deep_research",
            "strategy": "attachment_tool_research",
            "attachment_tool": tool_id,
            "composer_attachments": _attachments_telemetry(attachments),
        }
    if tool_id == "library":
        from core.app_settings import (
            external_knowledge_v2_enabled,
            internal_corpus_knowledge_enabled,
        )

        if external_knowledge_v2_enabled() and internal_corpus_knowledge_enabled():
            return {
                "route": "web",
                "strategy": "attachment_tool_library",
                "attachment_tool": tool_id,
                "composer_attachments": _attachments_telemetry(attachments),
            }
        return {
            "route": "rag",
            "strategy": "attachment_tool_library",
            "attachment_tool": tool_id,
            "composer_attachments": _attachments_telemetry(attachments),
        }
    if tool_id == "memory":
        return {
            "route": "memory",
            "strategy": "attachment_tool_memory",
            "attachment_tool": tool_id,
            "composer_attachments": _attachments_telemetry(attachments),
        }
    return None


def _attachments_telemetry(attachments: list[ComposerAttachment]) -> list[dict]:
    return [{"kind": a.kind, "id": a.id, "label": a.label} for a in attachments]


def build_referenced_conversation_context(
    session_id: str,
    db: DatabaseManager,
    *,
    max_chars: int = CONVERSATION_REF_BUDGET,
) -> str:
    """Load another session's transcript for injection into the current turn."""
    session = db.get_session(session_id)
    if not session:
        return ""
    title = str(session.get("title") or "Untitled").strip()
    messages = db.get_session_history(session_id)
    if not messages:
        return f"--- REFERENCED CONVERSATION: {title} ---\n(No messages in this conversation.)\n"

    lines = [f"--- REFERENCED CONVERSATION: {title} ---", ""]
    used = len(lines[0]) + 2
    for msg in messages:
        role = str(msg.get("role") or "user").lower()
        heading = _ROLE_HEADINGS.get(role, role.title())
        content = strip_tokens_for_display(str(msg.get("content") or "")).strip()
        if not content:
            continue
        block = f"{heading}: {content}"
        if used + len(block) + 2 > max_chars:
            lines.append("[…conversation truncated for context budget…]")
            break
        lines.append(block)
        lines.append("")
        used += len(block) + 2

    return "\n".join(lines).strip() + "\n"
