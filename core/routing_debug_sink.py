"""
Dedicated file sink for logger Qube.RoutingDebug (routing explainability JSON lines).
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

ROUTING_DEBUG_LOGGER_NAME = "Qube.RoutingDebug"

_HANDLER_ATTR = "_qube_routing_debug_rotating_sink"


def project_root() -> Path:
    """core/ -> repo root."""
    return Path(__file__).resolve().parent.parent


def default_routing_debug_log_path() -> Path:
    return project_root() / "logs" / "routing_debug.log"


def _ensure_logs_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def attach_routing_debug_file_sink(
    *,
    log_path: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Handler:
    """
    Attach a RotatingFileHandler to Qube.RoutingDebug. Idempotent.

    Returns the handler (existing or new). UTF-8 encoded.
    """
    path = Path(log_path) if log_path is not None else default_routing_debug_log_path()
    _ensure_logs_dir(path)

    lg = logging.getLogger(ROUTING_DEBUG_LOGGER_NAME)
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
    handler.setLevel(logging.DEBUG)
    lg.addHandler(handler)
    return handler


def detach_routing_debug_file_sink_for_tests() -> None:
    """Remove rotating sink(s) marked by us (tests only)."""
    lg = logging.getLogger(ROUTING_DEBUG_LOGGER_NAME)
    to_remove = [h for h in lg.handlers if getattr(h, _HANDLER_ATTR, False)]
    for h in to_remove:
        lg.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def quiet_routing_debug_logger_for_terminal() -> None:
    """
    Routing debug logs go only to the dedicated file handler(s); not stdout.
    """
    lg = logging.getLogger(ROUTING_DEBUG_LOGGER_NAME)
    lg.setLevel(logging.INFO)
    lg.propagate = False
