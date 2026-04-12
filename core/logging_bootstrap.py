"""
Application-wide logging helpers. LLM introspection logs use a dedicated file sink.
"""
from __future__ import annotations

import logging

from core.llm_debug_sink import attach_llm_debug_file_sink, quiet_llm_debug_logger_for_terminal

_LLM_DEBUG_INIT = False


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
