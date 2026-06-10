"""Cognition prompt adapter format inference."""
from __future__ import annotations

from core.cognition_prompt_adapter import (
    apply_qwen3_no_think_to_prompt,
    build_cognition_prompt,
    cognition_stop_tokens,
    finalize_cognition_system_prompt,
    infer_format_from_path,
    is_qwen3_cognition_model,
    resolve_cognition_chat_format,
)


def test_infer_phi_and_gemma():
    assert infer_format_from_path("/models/phi-3-mini.gguf") == "phi"
    assert infer_format_from_path("/models/gemma-2b-it.gguf") == "gemma"


def test_infer_qwen_chatml():
    assert infer_format_from_path("/models/qwen2.5-1.5b.gguf") == "chatml"
    assert infer_format_from_path("/models/Qwen3-1.7B-Q6_K.gguf") == "chatml"


def test_override_beats_inference():
    assert (
        resolve_cognition_chat_format("/models/phi-3.gguf", "chatml") == "chatml"
    )


def test_build_prompt_chatml_contains_im_start():
    p = build_cognition_prompt("sys", "usr", "chatml")
    assert "<|im_start|>system" in p
    assert cognition_stop_tokens("chatml")


def test_qwen3_detection():
    assert is_qwen3_cognition_model("/models/cognition/Qwen3-1.7B-Q6_K.gguf")
    assert not is_qwen3_cognition_model("/models/qwen2-0_5b-instruct-q4_k_m.gguf")


def test_finalize_system_appends_no_think_once():
    out = finalize_cognition_system_prompt(
        "Titling engine.", "/models/Qwen3-1.7B-Q6_K.gguf"
    )
    assert out.endswith("/no_think")
    again = finalize_cognition_system_prompt(out, "/models/Qwen3-1.7B-Q6_K.gguf")
    assert again.count("/no_think") == 1


def test_build_cognition_prompt_qwen3_includes_no_think():
    p = build_cognition_prompt(
        "Output only the title.",
        "hello",
        "chatml",
        model_path="/models/cognition/Qwen3-1.7B-Q6_K.gguf",
    )
    assert "/no_think" in p
    assert "<|im_start|>system" in p


def test_apply_no_think_to_flat_episode_style_prompt():
    flat = "You are writing a summary.\n\nCONVERSATION:\nhi"
    patched = apply_qwen3_no_think_to_prompt(
        flat, "/models/Qwen3-1.7B-Q6_K.gguf"
    )
    assert patched.startswith("/no_think")
