"""
Native-only sampling knobs for llama-cpp-python chat completions.

Stop strings are resolved by PromptContract and must remain format-specific.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("Qube.NativeLlamaInference")

def native_chat_completion_kwargs(llama: Any) -> dict[str, Any]:
    """
    Extra kwargs for Llama.create_completion on the native path (explicit prompt string).

    - repeat_penalty: chat handler defaults to 1.1; LM Studio / OpenAI-style APIs
      often behave closer to 1.0 for instruct models
    - top_p: handler default 0.95; a slightly tighter nucleus can reduce rambling / filler openers
      without changing user-facing temperature
    """
    return {
        "repeat_penalty": 1.0,
        "top_p": 0.92,
    }


def log_native_chat_debug(llama: Any, messages: list[dict]) -> None:
    """Lightweight summary: QUBE_LOG_NATIVE_CHAT=1. Full prompt + stops: use QUBE_LLM_DEBUG=1 (see native_llm_debug)."""
    if os.environ.get("QUBE_LOG_NATIVE_CHAT", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    try:
        roles = [m.get("role") for m in messages]
        nchars = sum(len(str(m.get("content", ""))) for m in messages)
        logger.info(
            "[Native][chat] format=%s messages=%d roles=%s total_chars=%d",
            getattr(llama, "chat_format", "?"),
            len(messages),
            roles,
            nchars,
        )
    except Exception as e:
        logger.debug("[Native][chat] log failed: %s", e)
