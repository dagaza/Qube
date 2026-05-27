"""Cognition prompt adapter format inference."""
from __future__ import annotations

from core.cognition_prompt_adapter import (
    build_cognition_prompt,
    cognition_stop_tokens,
    infer_format_from_path,
    resolve_cognition_chat_format,
)


def test_infer_phi_and_gemma():
    assert infer_format_from_path("/models/phi-3-mini.gguf") == "phi"
    assert infer_format_from_path("/models/gemma-2b-it.gguf") == "gemma"


def test_infer_qwen_chatml():
    assert infer_format_from_path("/models/qwen2.5-1.5b.gguf") == "chatml"


def test_override_beats_inference():
    assert (
        resolve_cognition_chat_format("/models/phi-3.gguf", "chatml") == "chatml"
    )


def test_build_prompt_chatml_contains_im_start():
    p = build_cognition_prompt("sys", "usr", "chatml")
    assert "<|im_start|>system" in p
    assert cognition_stop_tokens("chatml")
