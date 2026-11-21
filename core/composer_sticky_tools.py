"""Sticky per-chat composer knowledge routes (Web / Trusted / Evidence toggles)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StickyKnowledgeToolId = Literal["internet", "trusted", "evidence"]

STICKY_KNOWLEDGE_TOOL_IDS: frozenset[str] = frozenset({"internet", "trusted", "evidence"})

# Tool chips skipped when a sticky knowledge route already covers routing for the chat.
STICKY_COVERED_COMPOSER_TOOL_IDS: frozenset[str] = frozenset(
    {
        "internet",
        "trusted",
        "evidence",
        "research",
        "proresearch",
        "wikipedia",
        "pubmed",
        "arxiv",
        "science",
    }
)


def normalize_sticky_knowledge_tool(tool_id: str | None) -> str | None:
    tid = (tool_id or "").strip().lower()
    if tid in STICKY_KNOWLEDGE_TOOL_IDS:
        return tid
    return None


@dataclass(frozen=True)
class StickyKnowledgeWorkerFlags:
    sticky_knowledge_tool: str | None
    force_composer_tool: str | None
    force_web_enabled: bool


STICKY_KNOWLEDGE_UI_LABELS: dict[str, str] = {
    "internet": "Web",
    "evidence": "Evidence",
    "trusted": "Trusted",
    "research": "Deep Research",
}


def sticky_knowledge_ui_label(tool_id: str) -> str:
    tid = (tool_id or "").strip().lower()
    return STICKY_KNOWLEDGE_UI_LABELS.get(tid, tool_id or "Search")


def composer_sticky_knowledge_switched_request(
    *,
    previous_id: str,
    new_id: str,
):
    """In-app toast when the user enables a different sticky knowledge mode."""
    from core.app_notification_types import AppNotificationRequest
    from core.composer_draft import COMPOSER_ONE_SOURCE_DISMISS_MS

    prev = sticky_knowledge_ui_label(previous_id)
    new = sticky_knowledge_ui_label(new_id)
    return AppNotificationRequest(
        title="One search mode per chat",
        body=(
            f"Web, Evidence, Trusted, and Deep Research can't stay on together. "
            f"This chat now uses {new} instead of {prev}."
        ),
        auto_dismiss_ms=COMPOSER_ONE_SOURCE_DISMISS_MS,
        show_countdown=True,
        icon_name="fa5s.info-circle",
        severity="info",
        category="system",
        dedupe_key="composer_sticky_knowledge_exclusive",
    )


def sticky_knowledge_worker_flags(tool_id: str | None) -> StickyKnowledgeWorkerFlags:
    """Map a sticky UI knowledge tool id to worker force flags."""
    tid = normalize_sticky_knowledge_tool(tool_id)
    if tid == "internet":
        return StickyKnowledgeWorkerFlags(
            sticky_knowledge_tool="internet",
            force_composer_tool=None,
            force_web_enabled=True,
        )
    if tid in ("trusted", "evidence"):
        return StickyKnowledgeWorkerFlags(
            sticky_knowledge_tool=tid,
            force_composer_tool=tid,
            force_web_enabled=True,
        )
    return StickyKnowledgeWorkerFlags(
        sticky_knowledge_tool=None,
        force_composer_tool=None,
        force_web_enabled=False,
    )