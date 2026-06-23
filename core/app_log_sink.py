"""
Rotating file sink for general Qube application logs (Qube.*).

Default path: ~/.qube/logs/qube.log (see core.paths.logs_dir).

Specialized debug loggers (NativeLLM, RoutingDebug, SkillsDebug) are excluded —
they use dedicated files. Terminal output is unchanged.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from core.paths import logs_dir

APP_LOG_PREFIX = "Qube"

# Loggers with their own rotating files — do not duplicate into qube.log.
_EXCLUDED_LOGGER_NAMES = frozenset(
    {
        "Qube.NativeLLM.Debug",
        "Qube.RoutingDebug",
        "Qube.SkillsDebug",
    }
)

_HANDLER_ATTR = "_qube_app_rotating_sink"


class QubeAppLogFilter(logging.Filter):
    """Accept Qube.* records except dedicated debug loggers."""

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        if name != APP_LOG_PREFIX and not name.startswith(f"{APP_LOG_PREFIX}."):
            return False
        if name in _EXCLUDED_LOGGER_NAMES:
            return False
        for excluded in _EXCLUDED_LOGGER_NAMES:
            if name.startswith(f"{excluded}."):
                return False
        return True


def default_app_log_path() -> Path:
    return logs_dir() / "qube.log"


def app_log_enabled() -> bool:
    """False when ``QUBE_APP_LOG=0`` (or false/no/off). Enabled by default."""
    raw = os.environ.get("QUBE_APP_LOG", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def app_log_level() -> int:
    """File handler level from ``QUBE_APP_LOG_LEVEL`` (default INFO)."""
    raw = os.environ.get("QUBE_APP_LOG_LEVEL", "INFO").strip().upper()
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(raw, logging.INFO)


def _ensure_logs_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def attach_app_log_file_sink(
    *,
    log_path: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    level: int | None = None,
) -> logging.Handler | None:
    """
    Attach a filtered RotatingFileHandler to the root logger. Idempotent.

    Returns the handler (existing or new), or None when disabled via env.
    """
    if not app_log_enabled():
        return None

    env_path = (os.environ.get("QUBE_APP_LOG_FILE") or "").strip()
    if log_path is None and env_path:
        path = Path(env_path).expanduser()
    else:
        path = Path(log_path) if log_path is not None else default_app_log_path()
    _ensure_logs_dir(path)

    root = logging.getLogger()
    for h in root.handlers:
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
    handler.setLevel(level if level is not None else app_log_level())
    handler.addFilter(QubeAppLogFilter())
    root.addHandler(handler)
    return handler


def detach_app_log_file_sink_for_tests() -> None:
    """Remove rotating sink(s) marked by us (tests only)."""
    root = logging.getLogger()
    to_remove = [h for h in root.handlers if getattr(h, _HANDLER_ATTR, False)]
    for h in to_remove:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
