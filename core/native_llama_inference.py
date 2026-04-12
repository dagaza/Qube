"""
Native-only knobs for llama-cpp-python create_chat_completion.

LM Studio / llama.cpp server often merge extra stop strings so the model cannot
open a fake \"User:\" or next-turn block. The Jinja chat handler already appends
the GGUF eos_token to `stop`; we add conversational guards that are merged
(see llama_chat_format.chat_formatter_to_chat_completion_handler).
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("Qube.NativeLlamaInference")

# Merged with formatter stops — avoid overly short tokens that could clip legitimate prose.
_DEFAULT_EXTRA_STOPS: tuple[str, ...] = (
    "\n\nUser:",
    "\n\nUSER:",
    "\n\n### User",
    "\n\n### Instruction",
    "<|im_start|>user",
    "<|im_start|>user\n",
    "<|redacted_start_header_id|>user<|redacted_end_header_id|>",
    "<|User|>",
    "\n\nHuman:",
    "\n\n[INST]",
)


def native_chat_completion_kwargs(llama: Any) -> dict[str, Any]:
    """
    Extra kwargs for Llama.create_chat_completion on the native path.

    - stop: additional substrings that often mark the start of a spurious user turn
    - repeat_penalty: chat handler defaults to 1.1; LM Studio / OpenAI-style APIs
      often behave closer to 1.0 for instruct models
    - top_p: handler default 0.95; a slightly tighter nucleus can reduce rambling / filler openers
      without changing user-facing temperature
    """
    return {
        "stop": list(_DEFAULT_EXTRA_STOPS),
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
