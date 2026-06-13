"""Optional JSONL sink for skill activation telemetry."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from core.paths import logs_dir

SKILLS_DEBUG_LOGGER_NAME = "Qube.SkillsDebug"
_HANDLER_ATTR = "_qube_skills_debug_rotating_sink"


def default_skills_debug_log_path() -> Path:
    return logs_dir() / "skills_debug.log"


def attach_skills_debug_file_sink(
    *,
    log_path: Optional[Path] = None,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Handler:
    path = Path(log_path) if log_path is not None else default_skills_debug_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    lg = logging.getLogger(SKILLS_DEBUG_LOGGER_NAME)
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
    handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.setLevel(logging.INFO)
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return handler


def quiet_skills_debug_logger_for_terminal() -> None:
    """Skills debug logs go only to the dedicated file handler(s); not stdout."""
    lg = logging.getLogger(SKILLS_DEBUG_LOGGER_NAME)
    lg.setLevel(logging.INFO)
    lg.propagate = False


def log_skill_activation(payload: dict[str, Any]) -> None:
    lg = logging.getLogger(SKILLS_DEBUG_LOGGER_NAME)
    try:
        lg.info(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        pass
