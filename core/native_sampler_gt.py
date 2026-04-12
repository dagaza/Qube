"""
Ground-truth token IDs from llama-cpp-python's sampler (Llama.generate yield path).

Used only when QUBE_LLM_TOKEN_TRACE=1. Mirrors Llama._create_completion prompt assembly
for a plain string prompt with suffix=None so detokenize(prev_tokens=...) aligns with
inference (see upstream llama_cpp/llama.py).
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Callable, Optional

logger = logging.getLogger("Qube.NativeLLM.Debug")


def build_prompt_tokens_for_completion(llama: Any, prompt: str) -> list[int]:
    """
    Match Llama._create_completion assembly for prompt: str, suffix is None, echo False.
    Keep aligned with llama-cpp-python when changing versions.
    """
    if not isinstance(prompt, str):
        return []

    bos_token_id = int(llama.token_bos())
    try:
        cls_token_id = int(llama._model.token_cls())
    except Exception:
        cls_token_id = -1
    try:
        sep_token_id = int(llama._model.token_sep())
    except Exception:
        sep_token_id = -1

    prefix_token_id = 0
    suffix_token_id = 0
    suffix: Optional[str] = None

    md = getattr(llama, "metadata", None) or {}
    add_space_prefix = str(md.get("tokenizer.ggml.add_space_prefix", "true")).lower() == "true"
    if add_space_prefix and suffix_token_id >= 0 and suffix is not None:
        pass  # suffix None: no-op (mirrors upstream)

    bos_tokens: list[int] = [cls_token_id if cls_token_id != -1 else bos_token_id]
    eos_tokens: list[int] = [
        sep_token_id if sep_token_id != -1 else int(llama.token_eos())
    ]

    try:
        if (not llama._model.add_bos_token()) or bos_tokens[:1] == [-1]:
            bos_tokens = []
    except Exception:
        if bos_tokens[:1] == [-1]:
            bos_tokens = []

    try:
        if (not llama._model.add_eos_token()) and sep_token_id == -1:
            eos_tokens = []
    except Exception:
        pass

    special_flag = prefix_token_id < 0 or suffix is None
    prefix_tokens: list[int] = (
        ([prefix_token_id] if prefix_token_id >= 0 and suffix is not None else [])
        + (
            (
                llama.tokenize(
                    prompt.encode("utf-8"),
                    add_bos=False,
                    special=special_flag,
                )
            )
            if prompt != ""
            else []
        )
    )
    suffix_tokens: list[int] = []
    middle_tokens: list[int] = []

    spm_infill = bool(getattr(llama, "spm_infill", False))
    prompt_tokens: list[int] = (
        bos_tokens
        + (
            (suffix_tokens + prefix_tokens + middle_tokens)
            if spm_infill
            else (prefix_tokens + suffix_tokens + middle_tokens)
        )
        + eos_tokens
    )
    return prompt_tokens


def detokenize_sampler_token_ids(
    llama: Any, prompt_tokens: list[int], completion_ids: list[int]
) -> list[str]:
    """Per-step detokenize for each sampled id (matches llama.cpp stream pieces)."""
    out: list[str] = []
    for i, tid in enumerate(completion_ids):
        try:
            prev = prompt_tokens + completion_ids[:i]
            b = llama.detokenize([tid], prev_tokens=prev, special=False)
            out.append(b.decode("utf-8", errors="replace"))
        except Exception:
            out.append("")
    return out


def sequence_prefix_match_score(a: list[int], b: list[int]) -> float:
    """Share of the longer sequence matched by identical prefix (0..1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i / max(len(a), len(b), 1)
    return n / max(len(a), len(b), 1)


@contextmanager
def sampler_token_generate_capture(
    *,
    capture_ids: list[int],
    early_n: int,
    early_callback: Optional[Callable[[list[int]], None]],
):
    """
    Wrap llama_cpp.Llama.generate to record each yielded token id (sampler output).

    Restores the original method on exit. Safe for single-threaded NativeLlamaEngine.
    """
    try:
        from llama_cpp import Llama as LlamaCls
    except ImportError:
        yield
        return

    orig = LlamaCls.generate
    early_fired = False

    def wrapped_generate(self, *args: Any, **kwargs: Any):
        nonlocal early_fired
        for tok in orig(self, *args, **kwargs):
            tid = int(tok)
            capture_ids.append(tid)
            if (
                early_callback is not None
                and not early_fired
                and len(capture_ids) >= early_n
            ):
                early_fired = True
                try:
                    early_callback(capture_ids[:early_n])
                except Exception as e:
                    logger.debug("[sampler_gt] early callback failed: %s", e)
            yield tok

    LlamaCls.generate = wrapped_generate  # type: ignore[assignment]
    try:
        yield
    finally:
        LlamaCls.generate = orig  # type: ignore[assignment]
