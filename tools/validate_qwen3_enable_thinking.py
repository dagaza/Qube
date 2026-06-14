"""
Validate Qwen3 enable_thinking=False via chat_handler template kwargs.

Usage (from repo root):
  python -m tools.validate_qwen3_enable_thinking
  python -m tools.validate_qwen3_enable_thinking --model ~/.qube/models/cognition/Qwen3-1.7B-Q6_K.gguf
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any

logger = logging.getLogger("Qube.ValidateQwen3Thinking")

_THINK_TAG_RE = re.compile(r"(?is)<(?:redacted_)?think(?:ing)?>")


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_model(model_path: str, *, n_ctx: int = 4096, n_threads: int = 0):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit("llama-cpp-python is required") from exc

    from core.auxiliary_cognition import cognition_n_ctx_for_path
    from core.cognition_prompt_adapter import resolve_cognition_chat_format

    path = os.path.abspath(model_path)
    if not os.path.isfile(path):
        raise SystemExit(f"Model not found: {path}")
    ctx = cognition_n_ctx_for_path(path) if n_ctx <= 0 else n_ctx
    chat_format = resolve_cognition_chat_format(path)
    kwargs: dict[str, Any] = {
        "model_path": path,
        "n_gpu_layers": 0,
        "n_ctx": ctx,
        "verbose": False,
    }
    if n_threads > 0:
        kwargs["n_threads"] = n_threads
    return Llama(**kwargs), path, chat_format


def main() -> int:
    rr = _repo_root()
    if rr not in sys.path:
        sys.path.insert(0, rr)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | [%(name)s] %(message)s",
    )

    from core.auxiliary_cognition import resolve_active_cognition_path
    from core.qwen3_sidecar_inference import (
        chat_completion_complete,
        llama_cpp_supports_template_kwargs_via_handler,
    )

    p = argparse.ArgumentParser(description="Validate Qwen3 enable_thinking=False")
    p.add_argument("--model", default="", help="Path to Qwen3 GGUF")
    p.add_argument("--n-ctx", type=int, default=0)
    p.add_argument("--n-threads", type=int, default=0)
    p.add_argument("--json-out", default="", help="Optional JSON results path")
    args = p.parse_args()

    try:
        import llama_cpp

        llama_version = llama_cpp.__version__
    except ImportError:
        llama_version = "unknown"

    model_path = args.model or resolve_active_cognition_path()
    model, resolved_path, chat_format = _load_model(
        model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
    )

    messages = [
        {"role": "system", "content": "Reply with exactly the word TEST"},
        {"role": "user", "content": "Say TEST"},
    ]
    template_kwargs = {"enable_thinking": False}

    raw, diag = chat_completion_complete(
        model,
        messages,
        max_tokens=32,
        temperature=0.1,
        chat_template_kwargs=template_kwargs,
    )

    has_think_tags = bool(_THINK_TAG_RE.search(raw or ""))
    result = {
        "llama_cpp_version": llama_version,
        "model_path": resolved_path,
        "chat_format": chat_format,
        "handler_wrap_supported": llama_cpp_supports_template_kwargs_via_handler(),
        "chat_template_kwargs": template_kwargs,
        "raw_response": raw,
        "raw_response_length": len(raw or ""),
        "finish_reason": diag.finish_reason,
        "completion_tokens": diag.completion_tokens,
        "prompt_tokens": diag.prompt_tokens,
        "total_tokens": diag.total_tokens,
        "eos_encountered": diag.eos_encountered,
        "has_think_tags": has_think_tags,
        "generation_error": diag.generation_error,
        "enable_thinking_honored": not has_think_tags and not diag.generation_error,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("Wrote %s", args.json_out)

    if diag.generation_error:
        logger.error("Generation failed: %s", diag.generation_error)
        return 2
    if has_think_tags:
        logger.error("Think tags present despite enable_thinking=False")
        return 1
    logger.info("enable_thinking=False honored (no think tags)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
