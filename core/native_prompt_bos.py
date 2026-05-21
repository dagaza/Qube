"""
Strip duplicate leading BOS from rendered chat prompts before ``create_completion``.

GGUF Jinja templates often emit ``bos_token`` while llama.cpp also prepends BOS when
``add_bos_token()`` is enabled — llama-cpp-python warns when the assembled sequence
starts with two BOS ids.
"""
from __future__ import annotations

import logging
from typing import Any

from core.native_llm_debug import llama_eos_bos_strings
from core.native_sampler_gt import build_prompt_tokens_for_completion

logger = logging.getLogger("Qube.NativeLLM.Debug")

_MAX_STRIP_PASSES = 4


def has_duplicate_leading_bos(llama: Any, prompt: str) -> bool:
    """True when ``build_prompt_tokens_for_completion`` would start with BOS twice."""
    if not isinstance(prompt, str) or not prompt:
        return False
    try:
        bos_id = int(llama.token_bos())
    except Exception:
        return False
    tokens = build_prompt_tokens_for_completion(llama, prompt)
    return len(tokens) >= 2 and tokens[0] == bos_id and tokens[1] == bos_id


def _strip_one_leading_bos_text(llama: Any, prompt: str) -> tuple[str, bool]:
    _, bos_text = llama_eos_bos_strings(llama)
    if bos_text and prompt.startswith(bos_text):
        return prompt[len(bos_text) :], True
    try:
        bos_id = int(llama.token_bos())
        ids = llama.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
        if ids and ids[0] == bos_id:
            rest = llama.detokenize(ids[1:]).decode("utf-8", errors="replace")
            return rest, True
    except Exception:
        pass
    return prompt, False


def dedupe_leading_bos_for_completion(llama: Any, prompt: str) -> tuple[str, bool]:
    """
    Remove template-emitted leading BOS when llama.cpp will prepend BOS again.

    Returns ``(prompt, changed)``.
    """
    if not isinstance(prompt, str) or not prompt:
        return prompt, False

    p = prompt
    changed = False
    for _ in range(_MAX_STRIP_PASSES):
        if not has_duplicate_leading_bos(llama, p):
            break
        p, step = _strip_one_leading_bos_text(llama, p)
        if not step:
            break
        changed = True
    return p, changed


def prepare_completion_prompt(llama: Any, prompt: str) -> str:
    """Apply BOS dedupe; log once per changed prompt at debug level."""
    cleaned, changed = dedupe_leading_bos_for_completion(llama, prompt)
    if changed:
        logger.debug(
            "[Native] stripped duplicate leading BOS from completion prompt (chars %d -> %d)",
            len(prompt or ""),
            len(cleaned or ""),
        )
    return cleaned
