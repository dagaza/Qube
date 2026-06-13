"""Diagnostic log locations and helpers for Settings → Advanced."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from core.llm_debug_sink import default_llm_debug_log_path
from core.paths import logs_dir
from core.routing_debug_sink import default_routing_debug_log_path

logger = logging.getLogger("Qube.DiagnosticLogs")

PathFn = Callable[[], Path]


@dataclass(frozen=True)
class DiagnosticLogSpec:
    id: str
    title: str
    description: str
    path_fn: PathFn
    note: str = ""
    supports_recording_toggle: bool = False


DIAGNOSTIC_LOGS: tuple[DiagnosticLogSpec, ...] = (
    DiagnosticLogSpec(
        id="llm_debug",
        title="LLM debug log",
        description=(
            "Native LLM introspection: reconstructed prompts, completion output traces, "
            "discourse events, and optional token/causality JSON."
        ),
        path_fn=default_llm_debug_log_path,
    ),
    DiagnosticLogSpec(
        id="routing_debug",
        title="Routing debug log",
        description=(
            "Per-turn routing explainability as compact JSONL. Enable recording below "
            "to capture new chat turns; existing lines stay in the log file."
        ),
        path_fn=default_routing_debug_log_path,
        supports_recording_toggle=True,
    ),
)

_LOGS_BY_ID: dict[str, DiagnosticLogSpec] = {spec.id: spec for spec in DIAGNOSTIC_LOGS}


def get_diagnostic_log(log_id: str) -> DiagnosticLogSpec | None:
    return _LOGS_BY_ID.get(log_id)


def iter_diagnostic_logs() -> Iterable[DiagnosticLogSpec]:
    return DIAGNOSTIC_LOGS


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


def describe_log_status(spec: DiagnosticLogSpec) -> str:
    if spec.id == "routing_debug":
        return describe_routing_log_status(spec.path_fn())
    return describe_log_file(spec.path_fn())


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
