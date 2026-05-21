"""
Application-wide logging helpers. LLM introspection logs use a dedicated file sink.
"""
from __future__ import annotations

import logging

from core.llm_debug_sink import attach_llm_debug_file_sink, quiet_llm_debug_logger_for_terminal
from core.routing_debug_sink import (
    attach_routing_debug_file_sink,
    quiet_routing_debug_logger_for_terminal,
)

_LLM_DEBUG_INIT = False
_ROUTING_DEBUG_INIT = False


def init_llm_debug_logging() -> None:
    """
    Route Qube.NativeLLM.Debug to logs/llm_debug.log (rotating) and keep the terminal clean.

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
