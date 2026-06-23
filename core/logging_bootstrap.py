"""
Application-wide logging helpers. LLM introspection logs use a dedicated file sink.
"""
from __future__ import annotations

import logging

from core.app_log_sink import attach_app_log_file_sink, default_app_log_path
from core.llm_debug_sink import attach_llm_debug_file_sink, quiet_llm_debug_logger_for_terminal
from core.routing_debug_sink import (
    attach_routing_debug_file_sink,
    quiet_routing_debug_logger_for_terminal,
)
from core.skills.debug_sink import (
    attach_skills_debug_file_sink,
    quiet_skills_debug_logger_for_terminal,
)

_LLM_DEBUG_INIT = False
_ROUTING_DEBUG_INIT = False
_SKILLS_DEBUG_INIT = False
_APP_LOG_INIT = False


def init_llm_debug_logging() -> None:
    """
    Route Qube.NativeLLM.Debug to ~/.qube/logs/llm_debug.log (rotating) and keep the terminal clean.

    Safe to call multiple times (no duplicate file handlers). Call after logging.basicConfig
    if the root logger is already configured.
    """
    global _LLM_DEBUG_INIT
    if _LLM_DEBUG_INIT:
        return

    attach_llm_debug_file_sink()
    quiet_llm_debug_logger_for_terminal()
    _LLM_DEBUG_INIT = True


def llm_debug_logging_initialized() -> bool:
    return _LLM_DEBUG_INIT


def init_routing_debug_logging() -> None:
    """
    Route Qube.RoutingDebug to logs/routing_debug.log (rotating) and keep terminal clean.

    Safe to call multiple times (no duplicate file handlers). Call after logging.basicConfig
    if the root logger is already configured.
    """
    global _ROUTING_DEBUG_INIT
    if _ROUTING_DEBUG_INIT:
        return

    attach_routing_debug_file_sink()
    quiet_routing_debug_logger_for_terminal()
    _ROUTING_DEBUG_INIT = True


def routing_debug_logging_initialized() -> bool:
    return _ROUTING_DEBUG_INIT


def init_skills_debug_logging() -> None:
    """
    Route Qube.SkillsDebug to logs/skills_debug.log (rotating) and keep terminal clean.

    Safe to call multiple times (no duplicate file handlers). Call after logging.basicConfig
    if the root logger is already configured.
    """
    global _SKILLS_DEBUG_INIT
    if _SKILLS_DEBUG_INIT:
        return

    attach_skills_debug_file_sink()
    quiet_skills_debug_logger_for_terminal()
    _SKILLS_DEBUG_INIT = True


def skills_debug_logging_initialized() -> bool:
    return _SKILLS_DEBUG_INIT


def init_app_logging() -> None:
    """
    Route general ``Qube.*`` logs to ~/.qube/logs/qube.log (rotating, INFO by default).

    Terminal output is unchanged. Disable with ``QUBE_APP_LOG=0``; verbose file
    capture with ``QUBE_APP_LOG_LEVEL=DEBUG``.
    """
    global _APP_LOG_INIT
    if _APP_LOG_INIT:
        return

    handler = attach_app_log_file_sink()
    if handler is not None:
        logging.getLogger("Qube.Core").info(
            "App log file initialized at %s (level=%s)",
            default_app_log_path(),
            logging.getLevelName(handler.level),
        )
    _APP_LOG_INIT = True


def app_logging_initialized() -> bool:
    return _APP_LOG_INIT


def ensure_root_logging_minimal() -> None:
    """
    If the root logger has no handlers, attach a basic StreamHandler so other loggers work.
    main.py usually calls logging.basicConfig first; this is a no-op then.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
    )
