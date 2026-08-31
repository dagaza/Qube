"""Granular bootstrap / startup trace for field debugging (JSONL + optional console).

Enable with ``--bootstrap-trace`` or ``QUBE_BOOTSTRAP_TRACE=1``. On Windows, launch
from PowerShell to see live lines on stderr; the JSONL file is always written when
tracing is enabled (including Start-menu launches with no console).

Trace file: ``~/.qube/logs/bootstrap-trace.jsonl`` (Windows:
``%LOCALAPPDATA%\\Qube\\logs\\bootstrap-trace.jsonl``).
Latest snapshot: ``bootstrap-state.json`` in the same directory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("Qube.BootstrapTrace")

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_TRACE_FILE = "bootstrap-trace.jsonl"
_STATE_FILE = "bootstrap-state.json"
_ENABLED = False
_CONSOLE_HANDLER: logging.Handler | None = None


def bootstrap_trace_enabled() -> bool:
    return _ENABLED


def bootstrap_trace_path() -> Path:
    from core.paths import logs_dir

    return logs_dir() / _TRACE_FILE


def bootstrap_state_path() -> Path:
    from core.paths import logs_dir

    return logs_dir() / _STATE_FILE


def configure_bootstrap_trace(args: Any | None = None) -> None:
    """Activate tracing from ``--bootstrap-trace`` or ``QUBE_BOOTSTRAP_TRACE=1``."""
    global _ENABLED, _CONSOLE_HANDLER

    if args is not None and getattr(args, "bootstrap_trace", False):
        os.environ["QUBE_BOOTSTRAP_TRACE"] = "1"

    _ENABLED = os.environ.get("QUBE_BOOTSTRAP_TRACE", "").strip().lower() in _TRUTHY
    if not _ENABLED:
        return

    os.environ.setdefault("QUBE_APP_LOG_LEVEL", "DEBUG")

    if _CONSOLE_HANDLER is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="[bootstrap] %(message)s",
            )
        )
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        _CONSOLE_HANDLER = handler

    record_bootstrap_trace(
        "trace_enabled",
        argv=list(sys.argv),
        trace_file=str(bootstrap_trace_path()),
    )


def record_bootstrap_trace(event: str, **extra: Any) -> None:
    """Append one JSONL event and mirror a short line to the bootstrap console logger."""
    if not _ENABLED:
        return

    payload: dict[str, Any] = {
        "event": event,
        "timestamp": time.time(),
        "monotonic": round(time.monotonic(), 3),
    }
    if extra:
        payload.update(extra)

    trace_path = bootstrap_trace_path()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    bootstrap_state_path().write_text(
        json.dumps(payload, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if extra:
        logger.info("%s %s", event, json.dumps(extra, sort_keys=True, default=str))
    else:
        logger.info("%s", event)


def record_startup_progress(event: str, **extra: Any) -> None:
    """Record bootstrap progress for field trace and WinGet CI smoke (when active)."""
    record_bootstrap_trace(event, **extra)
    from core.winget_validation import is_winget_smoke_validation, record_boot_state

    if is_winget_smoke_validation():
        record_boot_state(event, **extra)


def reset_bootstrap_trace_for_tests() -> None:
    """Clear in-process trace state (unit tests only)."""
    global _ENABLED, _CONSOLE_HANDLER
    _ENABLED = False
    if _CONSOLE_HANDLER is not None:
        logger.removeHandler(_CONSOLE_HANDLER)
        _CONSOLE_HANDLER = None
    os.environ.pop("QUBE_BOOTSTRAP_TRACE", None)
