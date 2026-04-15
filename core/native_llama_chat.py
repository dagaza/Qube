"""
Helpers for llama-cpp-python chat completions: OpenAI-style message normalization
and aligning chat_format with GGUF tokenizer metadata (parity with LM Studio / server).
"""
from __future__ import annotations

import logging
import os
from typing import Any
from core.app_settings import get_internal_native_chat_format

logger = logging.getLogger("Qube.NativeLlamaChat")


def normalize_chat_messages(messages: list[dict]) -> list[dict]:
    """
    Produce strict {\"role\", \"content\"} dicts with string content for create_chat_completion.

    Merges consecutive system messages (same ordering as a single system prefix).
    """
    if not messages:
        return []

    normalized: list[dict] = []
    pending_system: list[str] = []

    def flush_system() -> None:
        nonlocal pending_system
        if not pending_system:
            return
        merged = "\n\n".join(s for s in pending_system if str(s).strip())
        pending_system = []
        if merged.strip():
            normalized.append({"role": "system", "content": merged})

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "user").strip().lower()
        if role not in ("system", "user", "assistant", "tool", "function"):
            role = "user"
        content = _stringify_message_content(m.get("content"))
        if role == "system":
            pending_system.append(content)
            continue
        flush_system()
        normalized.append({"role": role, "content": content})

    flush_system()  # trailing system-only prompts
    return normalized


def _stringify_message_content(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for part in raw:
            if isinstance(part, dict):
                t = part.get("type", "")
                if t == "text" and "text" in part:
                    parts.append(str(part["text"]))
                elif "text" in part:
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(raw)


def _model_identity_text(llama: Any) -> str:
    md = getattr(llama, "metadata", None) or {}
    names: list[str] = []
    if isinstance(md, dict):
        for k in ("general.name", "general.basename", "name"):
            v = md.get(k)
            if isinstance(v, str) and v.strip():
                names.append(v.strip())
    mp = getattr(llama, "model_path", None)
    if isinstance(mp, str) and mp:
        names.append(os.path.basename(mp))
    return " ".join(names).lower()


def prefer_gguf_jinja_chat_format(llama: Any) -> None:
    """
    If llama-cpp-python fell back to \"llama-2\" but the GGUF embeds tokenizer.chat_template,
    switch to the Jinja handler (\"chat_template.default\") so system prompts match LM Studio.

    Mutates llama.chat_format in place when safe.
    """
    try:
        # Respect explicit user choice. Auto mode is where model-aware coercion is allowed.
        if get_internal_native_chat_format() != "auto":
            return
        fmt = getattr(llama, "chat_format", None)
        md = getattr(llama, "metadata", None) or {}
        handlers = getattr(llama, "_chat_handlers", None) or {}
        ident = _model_identity_text(llama)
        is_nvidia_family = ("nemotron" in ident) or ("nvidia" in ident)
        if (
            is_nvidia_family
            and fmt == "llama-2"
            and "chatml" in handlers
        ):
            llama.chat_format = "chatml"
            logger.info(
                "[Native] chat_format adjusted: llama-2 -> chatml "
                "(NVIDIA/Nemotron auto-detect)"
            )
            return
        if (
            fmt == "llama-2"
            and "tokenizer.chat_template" in md
            and "chat_template.default" in handlers
        ):
            llama.chat_format = "chat_template.default"
            logger.info(
                "[Native] chat_format adjusted: llama-2 -> chat_template.default "
                "(GGUF contains tokenizer.chat_template)"
            )
    except Exception as e:
        logger.debug("[Native] prefer_gguf_jinja_chat_format: %s", e)
