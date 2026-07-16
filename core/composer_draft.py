"""Structured composer draft: chips in UI, tokens only at serialize time."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.composer_attachments import (
    COMPOSER_TOOLS,
    ComposerAttachment,
    format_token,
)
from core.composer_skills import (
    ComposerSkillMention,
    format_skill_token,
    parse_composer_input,
    skill_mention_from_id,
)

ROUTING_REJECT_ONE_SOURCE = "one_source_limit"
COMPOSER_ONE_SOURCE_DISMISS_MS = 5000


@dataclass
class ComposerDraft:
    """Plain body plus ordered routing attachments and skill mentions."""

    body: str = ""
    routing: list[ComposerAttachment] = field(default_factory=list)
    skills: list[ComposerSkillMention] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.body or "").strip() and not self.routing and not self.skills

    def routing_requires_body(self) -> bool:
        """True when tool/file chips are present but the composer body is empty."""
        if (self.body or "").strip():
            return False
        return any(att.kind in ("tool", "file") for att in self.routing)

    def clone(self) -> ComposerDraft:
        return ComposerDraft(
            body=self.body,
            routing=list(self.routing),
            skills=list(self.skills),
        )


def serialize_draft(draft: ComposerDraft) -> str:
    """Build the wire-format string stored in DB and passed as persist_content."""
    parts: list[str] = []
    for skill in draft.skills:
        parts.append(format_skill_token(skill.id))
    for att in draft.routing:
        parts.append(format_token(att))
    body = (draft.body or "").strip()
    if body:
        parts.append(body)
    return " ".join(parts)


def draft_from_text(text: str) -> ComposerDraft:
    """Parse token-bearing composer/history text into structured draft fields."""
    clean, attachments, enforced = parse_composer_input(text or "")
    skills: list[ComposerSkillMention] = []
    for sid in enforced:
        mention = skill_mention_from_id(sid)
        if mention is not None:
            skills.append(mention)
        else:
            skills.append(ComposerSkillMention(id=sid, label=sid))
    return ComposerDraft(body=clean, routing=list(attachments), skills=skills)


def merge_drafts(
    base: ComposerDraft,
    lifted: ComposerDraft,
    *,
    skip_internet_when_web_active: bool = False,
) -> tuple[ComposerDraft, str | None]:
    """Merge token-lift results into an existing draft (dedupe, one routing source)."""
    routing = list(base.routing)
    reject_reason: str | None = None
    for att in lifted.routing:
        candidate = ComposerDraft(
            body=base.body,
            routing=routing,
            skills=list(base.skills),
        )
        updated, added, reason = add_routing_attachment(
            candidate,
            att,
            skip_internet_when_web_active=skip_internet_when_web_active,
        )
        if reason == ROUTING_REJECT_ONE_SOURCE:
            reject_reason = reason
        elif added:
            routing = list(updated.routing)
    skills: list[ComposerSkillMention] = list(base.skills)
    for skill in lifted.skills:
        skills = _add_skill(skills, skill)
    body = lifted.body if lifted.body else base.body
    return ComposerDraft(body=body, routing=routing, skills=skills), reject_reason


def add_routing_attachment(
    draft: ComposerDraft,
    attachment: ComposerAttachment,
    *,
    skip_internet_when_web_active: bool = False,
) -> tuple[ComposerDraft, bool, str | None]:
    """
    Return updated draft, whether the attachment was added, and optional reject reason.

    When ``skip_internet_when_web_active`` is True, Internet tool chips are
    skipped because session Web search already covers routing.
    """
    if (
        skip_internet_when_web_active
        and attachment.kind == "tool"
        and attachment.id in {"internet", "trusted", "evidence", "research", "wikipedia", "pubmed", "arxiv"}
    ):
        return draft, False, None
    key = (attachment.kind, attachment.id)
    for existing in draft.routing:
        if (existing.kind, existing.id) == key:
            return draft, False, None
    if draft.routing:
        return draft, False, ROUTING_REJECT_ONE_SOURCE
    routing = list(draft.routing)
    routing.append(attachment)
    return (
        ComposerDraft(body=draft.body, routing=routing, skills=list(draft.skills)),
        True,
        None,
    )


def add_skill(draft: ComposerDraft, mention: ComposerSkillMention) -> tuple[ComposerDraft, bool]:
    """Return updated draft and whether the skill was added."""
    for existing in draft.skills:
        if existing.id.lower() == mention.id.lower():
            return draft, False
    skills = _add_skill(list(draft.skills), mention)
    return ComposerDraft(body=draft.body, routing=list(draft.routing), skills=skills), True


def remove_routing_at(draft: ComposerDraft, index: int) -> ComposerDraft:
    routing = list(draft.routing)
    if 0 <= index < len(routing):
        routing.pop(index)
    return ComposerDraft(body=draft.body, routing=routing, skills=list(draft.skills))


def remove_skill_at(draft: ComposerDraft, index: int) -> ComposerDraft:
    skills = list(draft.skills)
    if 0 <= index < len(skills):
        skills.pop(index)
    return ComposerDraft(body=draft.body, routing=list(draft.routing), skills=skills)


def _add_skill(
    skills: list[ComposerSkillMention],
    mention: ComposerSkillMention,
) -> list[ComposerSkillMention]:
    for existing in skills:
        if existing.id.lower() == mention.id.lower():
            return skills
    skills.append(mention)
    return skills


def routing_chip_icon(attachment: ComposerAttachment) -> str:
    if attachment.kind == "file":
        return "fa5s.file-alt"
    if attachment.kind == "conversation":
        return "fa5s.comments"
    if attachment.kind == "tool":
        icons = {
            "trusted": "fa5s.shield-alt",
            "evidence": "fa5s.microscope",
            "research": "fa5s.search-plus",
            "internet": "fa5s.globe",
            "wikipedia": "fa5s.book-open",
            "pubmed": "fa5s.notes-medical",
            "arxiv": "fa5s.atom",
            "library": "fa5s.book",
            "memory": "fa5s.brain",
        }
        return icons.get(attachment.id, "fa5s.tools")
    return "fa5s.paperclip"


def routing_chip_tooltip(attachment: ComposerAttachment, *, is_primary: bool) -> str:
    _ = is_primary
    if attachment.kind == "file":
        return f"Search scoped to {attachment.label} for this message."
    if attachment.kind == "conversation":
        return (
            f"Inject transcript from “{attachment.label}” for this turn only."
        )
    tool = next((t for t in COMPOSER_TOOLS if t["id"] == attachment.id), None)
    desc = tool["description"] if tool else attachment.label
    return f"{desc} for this message."


def composer_one_source_limit_request():
    """In-app toast when a second knowledge source chip is rejected."""
    from core.app_notification_types import AppNotificationRequest

    return AppNotificationRequest(
        title="One knowledge source per message",
        body="Remove the current chip to switch. Multiple sources are coming soon.",
        auto_dismiss_ms=COMPOSER_ONE_SOURCE_DISMISS_MS,
        show_countdown=True,
        icon_name="fa5s.info-circle",
        severity="info",
        category="system",
        dedupe_key="composer_one_source_limit",
    )


def composer_prompt_required_request():
    """In-app toast when the user tries to send a routing chip without prompt text."""
    from core.app_notification_types import AppNotificationRequest

    return AppNotificationRequest(
        title="Add a question or instruction",
        body="Type what you want to search or ask, then send.",
        auto_dismiss_ms=COMPOSER_ONE_SOURCE_DISMISS_MS,
        show_countdown=True,
        icon_name="fa5s.info-circle",
        severity="info",
        category="system",
        dedupe_key="composer_prompt_required",
    )


def deep_research_unavailable_request():
    """In-app toast when @research is used but the background worker is missing."""
    from core.app_notification_types import AppNotificationRequest

    body = "Deep research is unavailable (background worker not running)."

    return AppNotificationRequest(
        title="Deep research unavailable",
        body=body,
        auto_dismiss_ms=COMPOSER_ONE_SOURCE_DISMISS_MS,
        show_countdown=True,
        icon_name="fa5s.info-circle",
        severity="info",
        category="system",
        dedupe_key="deep_research_unavailable",
    )


def skill_chip_tooltip(mention: ComposerSkillMention) -> str:
    from core.skills.registry import get_skill

    skill = get_skill(mention.id)
    if skill is None:
        return f"Skill guidance: {mention.label}"
    desc = (skill.description or "").strip()
    if desc:
        return f"{mention.label} — {desc}"
    return f"Skill guidance: {mention.label}"
