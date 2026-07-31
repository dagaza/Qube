"""Diagnostic log locations and helpers for Settings → Diagnostics / Privacy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Literal

DiagnosticLogCategory = Literal["audit", "technical"]

from core.app_log_sink import default_app_log_path
from core.llm_debug_sink import default_llm_debug_log_path
from core.paths import logs_dir
from core.routing_debug_sink import default_routing_debug_log_path
from core.skills.debug_sink import default_skills_debug_log_path
from core.web_search_audit_sink import default_web_search_audit_log_path

logger = logging.getLogger("Qube.DiagnosticLogs")

PathFn = Callable[[], Path]

_HANDLER_LOGGER_NAMES = (
    "Qube.NativeLLM.Debug",
    "Qube.RoutingDebug",
    "Qube.SkillsDebug",
    "Qube.WebSearchAudit",
)


@dataclass(frozen=True)
class ClearLogResult:
    success: bool
    detail: str


@dataclass(frozen=True)
class DiagnosticLogSpec:
    id: str
    title: str
    description: str
    path_fn: PathFn
    category: DiagnosticLogCategory = "technical"
    note: str = ""
    supports_recording_toggle: bool = False
    recording_toggle_label: str = ""
    supports_redaction_toggle: bool = False
    redaction_toggle_label: str = ""


DIAGNOSTIC_LOGS: tuple[DiagnosticLogSpec, ...] = (
    DiagnosticLogSpec(
        id="app_log",
        title="Application log",
        description=(
            "General runtime events: boot, voice capture, status transitions, model load, "
            "and errors from all Qube modules (INFO by default)."
        ),
        path_fn=default_app_log_path,
        note=(
            "Terminal output is unchanged when file logging is off. "
            "Verbose file capture: QUBE_APP_LOG_LEVEL=DEBUG."
        ),
        supports_recording_toggle=True,
        recording_toggle_label="Record application events to this log",
    ),
    DiagnosticLogSpec(
        id="llm_debug",
        title="LLM debug log",
        description=(
            "Native LLM introspection: reconstructed prompts, completion output traces, "
            "discourse events, and optional token/causality JSON."
        ),
        path_fn=default_llm_debug_log_path,
        category="audit",
        note=(
            "This toggle controls file recording only. Heavy native introspection may "
            "still run when QUBE_LLM_DEBUG is enabled at launch."
        ),
        supports_recording_toggle=True,
        recording_toggle_label="Record LLM debug output to this log",
    ),
    DiagnosticLogSpec(
        id="routing_debug",
        title="Routing debug log",
        description=(
            "Per-turn routing explainability as compact JSONL. Enable recording below "
            "to capture new chat turns; existing lines stay in the log file."
        ),
        path_fn=default_routing_debug_log_path,
        category="audit",
        supports_recording_toggle=True,
        recording_toggle_label="Record routing decisions to this log",
        supports_redaction_toggle=True,
        redaction_toggle_label="Hash user queries in this log",
    ),
    DiagnosticLogSpec(
        id="web_search_audit",
        title="Web search log",
        description=(
            "Structured audit of live web searches: trigger reason, query text (raw and "
            "resolved), result URLs, and relevance-gate outcomes. DuckDuckGo SERP snippets "
            "only — individual result pages are not fetched."
        ),
        path_fn=default_web_search_audit_log_path,
        category="audit",
        note=(
            "Privacy: enable **Hash queries and omit snippet bodies** below, or set "
            "QUBE_WEB_SEARCH_AUDIT_REDACT=1 at launch to hash queries and omit snippet "
            "bodies in the log file."
        ),
        supports_recording_toggle=True,
        recording_toggle_label="Record web searches to this log",
        supports_redaction_toggle=True,
        redaction_toggle_label="Hash queries and omit snippet bodies in this log",
    ),
    DiagnosticLogSpec(
        id="skills_debug",
        title="Skills debug log",
        description=(
            "Per-turn skill activation scores and prompt injection telemetry. "
            "Enable recording below, then send a chat message to capture entries."
        ),
        path_fn=default_skills_debug_log_path,
        note=(
            "Requires Skills to be enabled under AI & Models. With Skills off, no "
            "activation telemetry is produced regardless of this log toggle."
        ),
        supports_recording_toggle=True,
        recording_toggle_label="Record skill activation to this log",
    ),
)

_LOGS_BY_ID: dict[str, DiagnosticLogSpec] = {spec.id: spec for spec in DIAGNOSTIC_LOGS}


def get_diagnostic_log(log_id: str) -> DiagnosticLogSpec | None:
    return _LOGS_BY_ID.get(log_id)


def iter_diagnostic_logs() -> Iterable[DiagnosticLogSpec]:
    return DIAGNOSTIC_LOGS


def iter_diagnostic_logs_by_category(
    category: DiagnosticLogCategory,
) -> Iterable[DiagnosticLogSpec]:
    return (spec for spec in DIAGNOSTIC_LOGS if spec.category == category)


def read_log_tail(path: Path, *, max_lines: int = 500) -> str:
    if max_lines <= 0:
        return ""
    if not path.is_file():
        return f"(no file yet: {path})"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(read error: {exc})"
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    omitted = len(lines) - max_lines
    body = "\n".join(lines[-max_lines:])
    return f"(… {omitted} earlier line(s) omitted …)\n{body}"


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def describe_log_file(path: Path) -> str:
    if not path.is_file():
        return "Not created yet"
    try:
        stat = path.stat()
    except OSError:
        return "Unavailable"
    updated = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{_format_bytes(stat.st_size)} · updated {updated}"


def describe_routing_log_status(path: Path) -> str:
    from core.app_settings import get_routing_debug_log_enabled
    from mcp.routing_debug import routing_debug_log_env_override, routing_debug_log_enabled

    if routing_debug_log_env_override() is not None:
        recording = "Recording on" if routing_debug_log_enabled() else "Recording off"
        recording += " (launch setting)"
    else:
        recording = "Recording on" if get_routing_debug_log_enabled() else "Recording off"
    return f"{recording} · {describe_log_file(path)}"


def describe_web_search_audit_log_status(path: Path) -> str:
    from core.app_settings import get_web_search_audit_log_enabled
    from core.web_search_audit import (
        web_search_audit_log_env_override,
        web_search_audit_log_enabled,
    )

    if web_search_audit_log_env_override() is not None:
        recording = "Recording on" if web_search_audit_log_enabled() else "Recording off"
        recording += " (launch setting)"
    else:
        recording = "Recording on" if get_web_search_audit_log_enabled() else "Recording off"
    return f"{recording} · {describe_log_file(path)}"


def describe_skills_log_status(path: Path) -> str:
    from core.app_settings import get_skills_debug_log_enabled

    recording = "Recording on" if get_skills_debug_log_enabled() else "Recording off"
    return f"{recording} · {describe_log_file(path)}"


def describe_app_log_status(path: Path) -> str:
    from core.app_log_sink import app_log_env_override
    from core.logging_bootstrap import effective_app_log_file_enabled

    override = app_log_env_override()
    if override is not None:
        recording = "Recording on" if override else "Recording off"
        recording += " (launch setting)"
    else:
        recording = "Recording on" if effective_app_log_file_enabled() else "Recording off"
    return f"{recording} · {describe_log_file(path)}"


def describe_llm_debug_log_status(path: Path) -> str:
    from core.llm_debug_sink import llm_debug_log_env_override
    from core.logging_bootstrap import effective_llm_debug_file_enabled

    if llm_debug_log_env_override() is not None:
        recording = "Recording on" if effective_llm_debug_file_enabled() else "Recording off"
        recording += " (launch setting)"
    else:
        recording = (
            "Recording on" if effective_llm_debug_file_enabled() else "Recording off"
        )
    return f"{recording} · {describe_log_file(path)}"


def diagnostic_log_recording_enabled(log_id: str) -> bool:
    if log_id == "routing_debug":
        from mcp.routing_debug import routing_debug_log_enabled

        return routing_debug_log_enabled()
    if log_id == "web_search_audit":
        from core.web_search_audit import web_search_audit_log_enabled

        return web_search_audit_log_enabled()
    if log_id == "skills_debug":
        from core.app_settings import get_skills_debug_log_enabled

        return get_skills_debug_log_enabled()
    if log_id == "app_log":
        from core.logging_bootstrap import effective_app_log_file_enabled

        return effective_app_log_file_enabled()
    if log_id == "llm_debug":
        from core.logging_bootstrap import effective_llm_debug_file_enabled

        return effective_llm_debug_file_enabled()
    return False


def describe_log_status(spec: DiagnosticLogSpec) -> str:
    if spec.id == "routing_debug":
        return describe_routing_log_status(spec.path_fn())
    if spec.id == "web_search_audit":
        return describe_web_search_audit_log_status(spec.path_fn())
    if spec.id == "skills_debug":
        return describe_skills_log_status(spec.path_fn())
    if spec.id == "app_log":
        return describe_app_log_status(spec.path_fn())
    if spec.id == "llm_debug":
        return describe_llm_debug_log_status(spec.path_fn())
    return describe_log_file(spec.path_fn())


def _find_file_handler(path: Path) -> logging.Handler | None:
    target = path.resolve()
    handlers: list[logging.Handler] = list(logging.getLogger().handlers)
    for name in _HANDLER_LOGGER_NAMES:
        handlers.extend(logging.getLogger(name).handlers)
    for handler in handlers:
        base = getattr(handler, "baseFilename", None)
        if not base:
            continue
        try:
            if Path(base).resolve() == target:
                return handler
        except OSError:
            continue
    return None


def _truncate_handler_stream(handler: logging.Handler) -> bool:
    stream = getattr(handler, "stream", None)
    if stream is None:
        return False
    if hasattr(handler, "acquire"):
        handler.acquire()
    try:
        stream.seek(0)
        stream.truncate(0)
        if hasattr(stream, "flush"):
            stream.flush()
        return True
    finally:
        if hasattr(handler, "release"):
            handler.release()


def _truncate_log_file(path: Path) -> bool:
    handler = _find_file_handler(path)
    if handler is not None and _truncate_handler_stream(handler):
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8"):
            pass
        return True
    except OSError as exc:
        logger.warning("Could not truncate diagnostic log %s: %s", path, exc)
        return False


def _delete_rotation_backups(path: Path) -> int:
    deleted = 0
    for backup in path.parent.glob(f"{path.name}.*"):
        if not backup.is_file() or backup == path:
            continue
        try:
            backup.unlink()
            deleted += 1
        except OSError as exc:
            logger.warning("Could not delete log backup %s: %s", backup, exc)
    return deleted


def clear_diagnostic_log(spec: DiagnosticLogSpec) -> ClearLogResult:
    """Clear log contents and rotated backups. Safe while handlers are attached."""
    path = spec.path_fn()
    existed = path.is_file()
    if not _truncate_log_file(path):
        return ClearLogResult(
            success=False,
            detail=f"Could not clear {path}.",
        )
    backups_removed = _delete_rotation_backups(path)
    if existed or backups_removed:
        detail = f"Cleared {spec.title}."
        if backups_removed:
            detail += f" Removed {backups_removed} rotated backup file(s)."
    else:
        detail = f"{spec.title} was already empty."
    return ClearLogResult(success=True, detail=detail)


def open_path_in_system(path: Path) -> bool:
    try:
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
    except ImportError:
        logger.warning("PyQt6 unavailable; cannot open path %s", path)
        return False

    target = path.resolve()
    if not target.exists():
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)
        except OSError as exc:
            logger.warning("Could not create log file before opening: %s", exc)
            return False
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


def open_logs_folder() -> bool:
    return open_path_in_system(logs_dir())
