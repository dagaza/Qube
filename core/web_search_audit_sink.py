"""
Dedicated file sink for logger Qube.WebSearchAudit (web search JSONL audit).
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from core.paths import logs_dir

WEB_SEARCH_AUDIT_LOGGER_NAME = "Qube.WebSearchAudit"

_HANDLER_ATTR = "_qube_web_search_audit_rotating_sink"


def default_web_search_audit_log_path() -> Path:
    return logs_dir() / "web_search.log"


def _ensure_logs_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def attach_web_search_audit_file_sink(
    *,
    log_path: Optional[Path] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Handler:
    """Attach a RotatingFileHandler to Qube.WebSearchAudit. Idempotent."""
    path = Path(log_path) if log_path is not None else default_web_search_audit_log_path()
    _ensure_logs_dir(path)

    lg = logging.getLogger(WEB_SEARCH_AUDIT_LOGGER_NAME)
    for h in lg.handlers:
        if getattr(h, _HANDLER_ATTR, False):
            return h

    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=False,
    )
    setattr(handler, _HANDLER_ATTR, True)
    handler.setFormatter(_make_formatter())
    handler.setLevel(logging.INFO)
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    return handler


def detach_web_search_audit_file_sink_for_tests() -> None:
    """Remove rotating sink(s) marked by us (tests only)."""
    lg = logging.getLogger(WEB_SEARCH_AUDIT_LOGGER_NAME)
    to_remove = [h for h in lg.handlers if getattr(h, _HANDLER_ATTR, False)]
    for h in to_remove:
        lg.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def quiet_web_search_audit_logger_for_terminal() -> None:
    """Web search audit logs go only to the dedicated file handler(s); not stdout."""
    lg = logging.getLogger(WEB_SEARCH_AUDIT_LOGGER_NAME)
    lg.setLevel(logging.INFO)
    lg.propagate = False
