"""Structured notification events for tray, in-app toasts, and OS notifications."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from core.app_notification_types import AppNotificationRequest


class NotificationSeverity(str, Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


NotificationCategory = Literal[
    "voice",
    "turn",
    "tool",
    "system",
    "background",
    "memory",
    "update",
]


@dataclass(frozen=True)
class NotificationAction:
    label: str
    action_id: str


@dataclass
class NotificationEvent:
    """Canonical notification payload consumed by NotificationService."""

    title: str
    body: str
    severity: NotificationSeverity = NotificationSeverity.INFO
    category: NotificationCategory = "system"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_label: str | None = None
    action_id: str | None = None
    auto_dismiss_ms: int = 0
    dedupe_key: str | None = None
    coalesce_group: str | None = None
    rate_limit_key: str | None = None
    rate_limit_sec: float = 0.0
    tray_bump: bool = False
    timestamp: float = field(default_factory=time.time)
    show_preview: bool = False
    show_countdown: bool = False
    icon_name: str | None = None

    def to_app_request(self) -> AppNotificationRequest:
        return AppNotificationRequest(
            title=self.title,
            body=self.body,
            action_label=self.action_label,
            action_id=self.action_id,
            auto_dismiss_ms=self.auto_dismiss_ms,
            show_countdown=self.show_countdown,
            icon_name=self.icon_name,
            severity=self.severity.value,
            category=self.category,
            event_id=self.event_id,
            dedupe_key=self.dedupe_key,
        )


# Well-known event builders
def turn_complete_event(*, session_id: str, preview: str = "") -> NotificationEvent:
    body = preview.strip() if preview else "Your assistant finished responding."
    return NotificationEvent(
        title="Reply ready",
        body=body,
        severity=NotificationSeverity.SUCCESS,
        category="turn",
        action_label="Open Qube",
        action_id="open_main_window",
        auto_dismiss_ms=5000,
        dedupe_key=f"turn_complete:{session_id}",
        rate_limit_key="turn_complete",
        rate_limit_sec=0.0,
    )


def stt_failed_event(*, reason: str = "") -> NotificationEvent:
    body = reason.strip() or "I didn't catch that — try again."
    return NotificationEvent(
        title="Voice input",
        body=body,
        severity=NotificationSeverity.WARNING,
        category="voice",
        action_label="Open Qube",
        action_id="open_main_window",
        auto_dismiss_ms=6000,
        dedupe_key="stt_failed",
        rate_limit_key="stt_failed",
        rate_limit_sec=10.0,
        tray_bump=True,
    )


VOICE_INPUT_UNAVAILABLE_BODY = (
    "Voice input unavailable — another application may be using your microphone. "
    "You may still use the text-based mode for interaction with your assistant."
)


def voice_input_unavailable_event() -> NotificationEvent:
    """Tier-A reactive notice when mic capture cannot open (e.g. call app holds the device)."""
    return NotificationEvent(
        title="Voice input unavailable",
        body=VOICE_INPUT_UNAVAILABLE_BODY,
        severity=NotificationSeverity.WARNING,
        category="voice",
        auto_dismiss_ms=0,
        dedupe_key="voice_input_unavailable",
        rate_limit_key="voice_input_unavailable",
        rate_limit_sec=60.0,
        tray_bump=True,
    )


def mic_error_event(*, detail: str = "") -> NotificationEvent:
    """Legacy builder — prefer ``voice_input_unavailable_event`` for mic open failures."""
    body = detail.strip() or "Microphone unavailable — check audio settings."
    return NotificationEvent(
        title="Microphone",
        body=body,
        severity=NotificationSeverity.WARNING,
        category="voice",
        action_label="Open Settings",
        action_id="open_settings",
        auto_dismiss_ms=0,
        dedupe_key="mic_error",
        rate_limit_key="mic_error",
        rate_limit_sec=60.0,
        tray_bump=True,
    )


def native_model_reloaded_from_settings_event(
    *,
    model_name: str,
    cpu_fallback: bool = False,
) -> NotificationEvent:
    """Toast when a settings-driven native reload succeeds (GPU/CPU/context)."""
    name = str(model_name or "").strip() or "Model"
    if cpu_fallback:
        body = (
            f"{name} is ready on CPU — GPU offload did not fit in memory. "
            "Lower GPU layers in Settings → AI & Models if you want to try again."
        )
    else:
        body = (
            f"{name} is ready with your updated hardware settings "
            "(GPU layers, CPU threads, or context limit)."
        )
    return NotificationEvent(
        title="Model reloaded",
        body=body,
        severity=NotificationSeverity.SUCCESS,
        category="system",
        auto_dismiss_ms=6000,
        dedupe_key=f"native_hardware_reload:{name}",
        rate_limit_key="native_hardware_reload",
        rate_limit_sec=3.0,
        icon_name="fa5s.cube",
    )


def needs_model_event() -> NotificationEvent:
    from core.app_settings import get_engine_mode
    from core.local_gguf_library import has_local_gguf_models

    dismiss_ms = 5000
    has_local = get_engine_mode() == "internal" and has_local_gguf_models()
    if has_local:
        return NotificationEvent(
            title="Model required",
            body="Pick a downloaded model from Select AI Model in the toolbar.",
            severity=NotificationSeverity.CRITICAL,
            category="system",
            action_label="Select AI Model",
            action_id="open_local_model_picker",
            auto_dismiss_ms=dismiss_ms,
            show_countdown=True,
            icon_name="fa5s.cube",
            dedupe_key="needs_model",
            tray_bump=True,
        )
    return NotificationEvent(
        title="Model required",
        body="Load a model to start chatting.",
        severity=NotificationSeverity.CRITICAL,
        category="system",
        action_label="Open Models",
        action_id="open_models",
        auto_dismiss_ms=dismiss_ms,
        show_countdown=True,
        icon_name="fa5s.cube",
        dedupe_key="needs_model",
        tray_bump=True,
    )


def ingestion_complete_event(*, file_count: int) -> NotificationEvent:
    noun = "document" if file_count == 1 else "documents"
    return NotificationEvent(
        title="Library updated",
        body=f"{file_count} {noun} added to your library.",
        severity=NotificationSeverity.SUCCESS,
        category="background",
        action_label="Open Library",
        action_id="open_library",
        auto_dismiss_ms=5000,
        coalesce_group="ingestion_complete",
        dedupe_key=f"ingestion_complete:{file_count}",
    )


def output_truncated_max_tokens_event(*, session_id: str) -> NotificationEvent:
    return NotificationEvent(
        title="Response truncated",
        body=(
            "The model reached its maximum length for this reply. "
            "The answer may be incomplete — try a shorter question or ask for a specific section."
        ),
        severity=NotificationSeverity.WARNING,
        category="turn",
        auto_dismiss_ms=10_000,
        dedupe_key=f"max_tokens:{session_id}",
        rate_limit_key="max_tokens_truncated",
        rate_limit_sec=5.0,
    )


def format_retry_in_progress_event(
    *,
    session_id: str,
    issues: list[str] | None = None,
) -> NotificationEvent:
    issue_set = {str(i) for i in (issues or [])}
    if "degeneration" in issue_set:
        detail = (
            "Output quality checks flagged possible repetition or formatting issues."
        )
    elif "template_leakage" in issue_set:
        detail = "Template artifacts were detected in the raw output."
    elif "role_confusion" in issue_set:
        detail = "The reply looked like a dialog leak rather than a single answer."
    elif "meta_preamble" in issue_set:
        detail = "The reply looked like internal planning text rather than a final answer."
    else:
        detail = "Output quality checks flagged a formatting issue."
    return NotificationEvent(
        title="Improving response",
        body=(
            f"Qube is re-generating this answer for better formatting. {detail} "
            "This may take a minute."
        ),
        severity=NotificationSeverity.WARNING,
        category="turn",
        auto_dismiss_ms=12_000,
        dedupe_key=f"format_retry:{session_id}",
        rate_limit_key="format_retry_in_progress",
        rate_limit_sec=5.0,
    )


def enrichment_complete_event(*, session_id: str, facts_stored: int) -> NotificationEvent:
    if facts_stored <= 0:
        return NotificationEvent(
            title="Memory",
            body="Memory extraction finished with no new facts.",
            severity=NotificationSeverity.INFO,
            category="memory",
            auto_dismiss_ms=4000,
            dedupe_key=f"enrichment:{session_id}:0",
        )
    noun = "memory" if facts_stored == 1 else "memories"
    return NotificationEvent(
        title="Memories saved",
        body=f"Saved {facts_stored} {noun} from your chat.",
        severity=NotificationSeverity.SUCCESS,
        category="memory",
        action_label="View Memories",
        action_id="open_memories",
        auto_dismiss_ms=8000,
        dedupe_key=f"enrichment:{session_id}:{facts_stored}",
        coalesce_group="enrichment_complete",
    )


def auto_backup_complete_event(*, destination: Path | str) -> NotificationEvent:
    path = Path(destination)
    return NotificationEvent(
        title="Local backup saved",
        body=f"Automatic backup saved to {path.name}.",
        severity=NotificationSeverity.SUCCESS,
        category="background",
        auto_dismiss_ms=8000,
        dedupe_key="auto_backup_complete",
        rate_limit_key="auto_backup_complete",
        rate_limit_sec=300.0,
    )


def auto_backup_failed_event(*, error: str = "") -> NotificationEvent:
    detail = (error or "").strip() or "Automatic backup could not be completed."
    return NotificationEvent(
        title="Automatic backup failed",
        body=detail,
        severity=NotificationSeverity.WARNING,
        category="background",
        auto_dismiss_ms=12_000,
        dedupe_key="auto_backup_failed",
        rate_limit_key="auto_backup_failed",
        rate_limit_sec=300.0,
        tray_bump=True,
    )


def deep_research_complete_event(
    *,
    session_id: str,
    query: str,
    source_count: int,
    synthesis_applied: bool = False,
) -> NotificationEvent:
    preview = (query or "").strip()
    if len(preview) > 72:
        preview = preview[:69] + "…"
    detail = f"{source_count} source(s)"
    if synthesis_applied:
        detail = f"synthesized report · {detail}"
    body = f"{preview} — {detail}" if preview else detail
    return NotificationEvent(
        title="Deep research complete",
        body=body,
        severity=NotificationSeverity.SUCCESS,
        category="system",
        auto_dismiss_ms=8000,
        dedupe_key=f"deep_research:{session_id}:{preview}",
        coalesce_group="deep_research_complete",
    )


def provider_limit_notification_event(event: object) -> NotificationEvent:
    """Informational notice when an anonymous provider quota is exhausted."""
    from core.knowledge.provider_credentials import get_provider_credential_spec

    provider_id = getattr(event, "provider_id", "")
    spec = get_provider_credential_spec(str(provider_id))
    label = spec.label if spec is not None else str(provider_id or "Provider")
    body = (
        f"You've reached the anonymous limit for {label} today. "
        "Add a free API key in Settings → Live sources for a higher daily budget, "
        "or try again after midnight UTC."
    )
    return NotificationEvent(
        title=f"{label} quota reached",
        body=body,
        severity=NotificationSeverity.INFO,
        category="system",
        action_label="Open Settings",
        action_id=f"open_settings_knowledge_credentials:{provider_id}",
        auto_dismiss_ms=12000,
        dedupe_key=f"provider_limit:{provider_id}",
        rate_limit_key=f"provider_limit:{provider_id}",
        rate_limit_sec=86400.0,
    )


def ddg_backoff_event(*, remaining_seconds: int = 0) -> NotificationEvent:
    """Warn when DuckDuckGo is paused after a bot challenge."""
    from core.knowledge.discovery.backoff import ddg_bot_backoff_seconds

    total_seconds = ddg_bot_backoff_seconds()
    minutes = max(1, (total_seconds + 59) // 60)
    body = (
        "DuckDuckGo returned a bot challenge, so DDG searches are paused for "
        f"{minutes} minutes to avoid further blocks. "
        "Web searches will continue using Brave or Wikipedia fallbacks when available."
    )
    if remaining_seconds > 0:
        remaining_minutes = max(1, (remaining_seconds + 59) // 60)
        body = (
            f"DuckDuckGo returned a bot challenge. DDG searches are paused for "
            f"about {remaining_minutes} more minutes "
            f"(~{minutes} min total) to avoid further blocks. "
            "Web searches will continue using Brave or Wikipedia fallbacks when available."
        )
    return NotificationEvent(
        title="DuckDuckGo search paused",
        body=body,
        severity=NotificationSeverity.WARNING,
        category="tool",
        action_label="Discovery settings",
        action_id="open_settings_knowledge_web_discovery",
        auto_dismiss_ms=15000,
        dedupe_key="ddg_backoff",
        rate_limit_key="ddg_backoff",
        rate_limit_sec=300.0,
        tray_bump=True,
    )


def discovery_tier_b_suggestion_event() -> NotificationEvent:
    """Suggest optional API fallback after repeated DDG challenges (private tier)."""
    return NotificationEvent(
        title="Repeated search blocks detected",
        body=(
            "DuckDuckGo has challenged several searches in the last 24 hours. "
            "Consider enabling “Private + API fallback” in Settings → Knowledge → "
            "Web search discovery and adding a free Brave Search API key for "
            "better reliability — DDG stays primary when it works."
        ),
        severity=NotificationSeverity.INFO,
        category="tool",
        action_label="Discovery settings",
        action_id="open_settings_knowledge_web_discovery",
        auto_dismiss_ms=20000,
        dedupe_key="discovery_tier_b_suggestion",
        rate_limit_key="discovery_tier_b_suggestion",
        rate_limit_sec=86400.0,
        tray_bump=True,
    )
